import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
import torch
import requests
from io import BytesIO
from PIL import Image
import time
import json
import asyncio
import os
import base64
from datetime import datetime, timezone
import threading
import gc
import argparse

app = FastAPI()

# Define pixel constraints for the model
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2-VL-7B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
)

processor = AutoProcessor.from_pretrained(model_name)

# Define request models according to Ollama's API specifications
class GenerateOptions(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None

class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None  # Images can be base64 strings, file paths, or URLs

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    options: Optional[GenerateOptions] = None
    stream: Optional[bool] = True  # Add stream parameter
    format: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    system: Optional[str] = None

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[GenerateOptions] = None
    system: Optional[str] = None
    images: Optional[List[str]] = None  # Images can be base64 strings, file paths, or URLs
    stream: Optional[bool] = True  # Add stream parameter
    format: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    system: Optional[str] = None

# Function to process and resize images
def process_images(image_data_list):
    # The max input image size for the Qwen2-VL-7B-Instruct model can be adjusted using pixel count limits.
    # The default range of visual tokens per image is 4-16,384, 
    # You can set the number of pixels between 
    # min_pixels = 256 * 28 * 28
    # max_pixels = 1280 * 28 * 28

    processed_images = []
    
    for image_data in image_data_list:
        image = None
        
        try:
            # Try to decode base64-encoded image data
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (base64.binascii.Error, IOError):
            pass  # Not base64 or invalid image, try next method
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

        if image is None:
            # Try to load from file path
            if os.path.exists(image_data):
                try:
                    image = Image.open(image_data).convert("RGB")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error loading image from path: {e}")
            else:
                # Try to load from URL
                try:
                    response = requests.get(image_data.strip())
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error loading image from URL: {e}")
        
        if image:
            # Get the current image size and calculate pixel count
            width, height = image.size
            current_pixel_count = width * height

            # Only resize if the image exceeds max pixel count or is too small
            if current_pixel_count > MAX_PIXELS or current_pixel_count < MIN_PIXELS:
                # Calculate the resizing factor while maintaining aspect ratio
                if current_pixel_count > MAX_PIXELS:
                    # Resize to fit within max_pixels
                    scale_factor = (MAX_PIXELS / current_pixel_count) ** 0.5
                else:
                    # Resize up to fit within min_pixels (if too small)
                    scale_factor = (MIN_PIXELS / current_pixel_count) ** 0.5

                # Calculate the new dimensions while keeping aspect ratio
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Resize the image maintaining the aspect ratio
                image = image.resize((new_width, new_height), Image.LANCZOS)

            processed_images.append(image)
        
    return processed_images

# Implement the /api/generate endpoint
@app.post("/api/generate")
async def generate(request: Request):
    req_data = await request.json()
    generate_request = GenerateRequest(**req_data)

    # Check if the requested model matches
    if generate_request.model != model_name:
        raise HTTPException(status_code=400, detail="Model not found")

    # Prepare the prompt and system prompt
    prompt = generate_request.prompt
    system_prompt = generate_request.system or None
    message_content = []

    # Handle images
    image_inputs = []
    if generate_request.images:
        image_inputs = process_images(generate_request.images)
        message_content.extend([{"type": "image", "image": img} for img in image_inputs])

    # Add text content
    if prompt:
        message_content.append({"type": "text", "text": prompt})

    # Prepare messages with optional system prompt
    messages = []
    if system_prompt:
        messages.insert(0, {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })
    messages.append({
        "role": "user",
        "content": message_content
    })

    # Prepare input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Set generation options
    generation_kwargs = {
        'max_new_tokens': generate_request.options.max_tokens if generate_request.options and generate_request.options.max_tokens else 128,
        'temperature': generate_request.options.temperature if generate_request.options and generate_request.options.temperature else 1.0,
        'top_p': generate_request.options.top_p if generate_request.options and generate_request.options.top_p else 1.0,
    }

    # Check if streaming is requested
    if generate_request.stream:
        # Real streaming response
        streamer = TextIteratorStreamer(
            tokenizer=processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs['streamer'] = streamer

        # Start the generation in a background thread
        def model_generate():
            with torch.no_grad():
                model.generate(**inputs, **generation_kwargs)

        generation_thread = threading.Thread(target=model_generate)
        generation_thread.start()

        # Prepare response according to Ollama's API
        async def response_generator():
            start_time = time.time()
            partial_response = ""

            for new_text in streamer:
                partial_response += new_text
                response_data = {
                    "model": generate_request.model,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec='microseconds'),
                    "response": new_text,
                    "done": False
                }
                yield json.dumps(response_data) + "\n"
                await asyncio.sleep(0)  # Yield control to the event loop

            generation_thread.join()
            end_time = time.time()

            # Final completion message
            response_data = {
                "model": generate_request.model,
                "created_at": datetime.now(timezone.utc).isoformat(timespec='microseconds'),
                "response": "",
                "done": True,
                "done_reason": "stop",
                "total_duration": int((end_time - start_time) * 1e9),  # Nanoseconds
                "eval_count": len(partial_response),  # Number of characters generated
            }
            yield json.dumps(response_data) + "\n"

            # Free up memory
            torch.cuda.empty_cache()
            gc.collect()

        return StreamingResponse(response_generator(), media_type="application/json")

    else:
        # Non-streaming response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Prepare response according to Ollama's API
        response_data = {
            "model": generate_request.model,
            "created_at": int(time.time()),
            "response": output_text,
            "done": True
        }

        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()
        return JSONResponse(content=response_data)

@app.post("/api/chat")
async def chat(request: Request):
    req_data = await request.json()
    chat_request = ChatRequest(**req_data)

    # Check if the requested model matches
    if chat_request.model != model_name:
        raise HTTPException(status_code=400, detail="Model not found")

    # Prepare the messages
    messages = []
    image_inputs = []

    for message in chat_request.messages:
        content = message.content
        message_content = []

        # Handle images from the 'images' field
        if message.images:
            image_inputs = process_images(message.images)
            message_content.extend([{"type": "image", "image": img} for img in image_inputs])

        # Add text content
        if content:
            message_content.append({"type": "text", "text": content})

        messages.append({
            "role": message.role,
            "content": message_content
        })

    if not chat_request.stream:
        # Implement 'system' argument logic
        if chat_request.system:
            # Add to context
            messages.insert(0, {
                'role': 'system',
                'content': [{'type': 'text', 'text': chat_request.system}]
            })

        # Implement 'tools' argument logic
        if chat_request.tools:
            tools_json = json.dumps(chat_request.tools)
            # Add to context
            messages.insert(0, {
                'role': 'system',
                'content': [{'type': 'tools', 'text': f"Use the following tools as needed to answer the question. If you call a tool, the answer will be returned to you for answering the user's request:\n{tools_json}"}]
            })

        # Implement 'format' argument logic
        if chat_request.format == "json":
            # search for system message and append instruction
            found = False
            for message in reversed(messages):
                if message['role'] == 'system':
                    found = True
                    message['content'].insert(0, {'type': 'text', 'text': 'You are a JSON-formatting assistant. Only output valid JSON and nothing else.'})
                    break

            if not found:
                # create new system message
                messages.insert(0, {
                    'role': 'system',
                    'content': [{'type': 'text', 'text': 'You are a JSON-formatting assistant. Only output valid JSON and nothing else.'}]
                })

    # Prepare input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Set generation options
    generation_kwargs = {
        'max_new_tokens': chat_request.options.max_tokens if chat_request.options and chat_request.options.max_tokens else 128,
        'temperature': chat_request.options.temperature if chat_request.options and chat_request.options.temperature else 1.0,
        'top_p': chat_request.options.top_p if chat_request.options and chat_request.options.top_p else 1.0,
    }

    # Check if streaming is requested
    if chat_request.stream:
        # Real streaming response
        streamer = TextIteratorStreamer(
            tokenizer=processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs['streamer'] = streamer

        # Start the generation in a background thread
        def generate():
            with torch.no_grad():
                model.generate(**inputs, **generation_kwargs)

        generation_thread = threading.Thread(target=generate)
        generation_thread.start()

        # Prepare response according to Ollama's API
        async def response_generator():
            start_time = time.time()
            partial_response = ""

            for new_text in streamer:
                partial_response += new_text
                response_data = {
                    "model": chat_request.model,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec='microseconds'),
                    "message": {
                        "role": "assistant",
                        "content": new_text
                    },
                    "done": False
                }
                yield json.dumps(response_data) + "\n"
                await asyncio.sleep(0)  # Yield control to the event loop

            generation_thread.join()
            end_time = time.time()

            # Final completion message
            response_data = {
                "model": chat_request.model,
                "created_at": datetime.now(timezone.utc).isoformat(timespec='microseconds'),
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done": True,
                "done_reason": "stop",
                "total_duration": int((end_time - start_time) * 1e9),  # Nanoseconds
                "eval_count": len(partial_response),  # Number of characters generated
            }
            yield json.dumps(response_data) + "\n"

            # Free up memory
            torch.cuda.empty_cache()
            gc.collect()

        return StreamingResponse(response_generator(), media_type="application/json")

    else:
        # Non-streaming response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Prepare response according to Ollama's API
        response_data = {
            "model": chat_request.model,
            "created_at": int(time.time()),
            "response": output_text,
            "done": True
        }

        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()

        return JSONResponse(content=response_data)

# Implement the /api/models endpoint
@app.get("/api/models")
async def list_models():
    # Return the available models
    return {"models": [model_name]}

if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    args = parser.parse_args()

    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)
