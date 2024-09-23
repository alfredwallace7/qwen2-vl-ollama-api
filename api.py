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

app = FastAPI()

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
    system_prompt: Optional[str] = None
    images: Optional[List[str]] = None  # Images can be base64 strings, file paths, or URLs
    stream: Optional[bool] = True  # Add stream parameter
    format: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    system: Optional[str] = None

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
    system_prompt = generate_request.system_prompt or None
    message_content = []

    if not generate_request.stream:
        # Implement 'format' argument logic
        if generate_request.format == "json":
            system_prompt = f"You are a JSON-formatting assistant. Only output valid JSON and nothing else."

        # Implement 'tools' argument logic
        if generate_request.tools:
            tools_json = json.dumps(generate_request.tools)
            if system_prompt:
                system_prompt += f"\nTools: {tools_json}"
            else:
                system_prompt = f"Tools: {tools_json}"

        # Implement 'system' argument logic
        if generate_request.system:
            if system_prompt:
                system_prompt += f"\nSystem: {generate_request.system}"
            else:
                system_prompt = f"System: {generate_request.system}"

    # Handle images
    if generate_request.images:
        for image_data in generate_request.images:
            image = None
            # Try to decode base64-encoded image data
            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                message_content.append({"type": "image", "image": image})
                continue  # Successfully loaded image, proceed to next
            except (base64.binascii.Error, IOError):
                pass  # Not base64 or invalid image, try next method

            # Try to load from file path
            if os.path.exists(image_data):
                try:
                    image = Image.open(image_data).convert("RGB")
                    message_content.append({"type": "image", "image": image})
                    continue
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error loading image from path: {e}")

            # Try to load from URL
            try:
                response = requests.get(image_data.strip())
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                message_content.append({"type": "image", "image": image})
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error loading image: {e}")

    # Add text content
    if prompt:
        message_content.append({"type": "text", "text": prompt})

    # Prepare messages with optional system prompt
    messages = []
    if system_prompt:
        messages.append({
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
    if generate_request.options and generate_request.options.stop:
        generation_kwargs['eos_token_id'] = processor.tokenizer.convert_tokens_to_ids(generate_request.options.stop)

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
                "load_duration": 0,  # Fill this based on actual load time if tracked
                "prompt_eval_duration": 0,  # Fill if needed
                "eval_count": len(partial_response),  # Number of characters generated
                "eval_duration": int((end_time - start_time) * 1e9)  # Placeholder for eval duration
            }
            yield json.dumps(response_data) + "\n"

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
        return JSONResponse(content=response_data)

# Implement the /api/chat endpoint
@app.post("/api/chat")
async def chat(request: Request):
    req_data = await request.json()
    chat_request = ChatRequest(**req_data)

    # Check if the requested model matches
    if chat_request.model != model_name:
        raise HTTPException(status_code=400, detail="Model not found")

    # Prepare the messages
    messages = []

    for message in chat_request.messages:
        content = message.content
        message_content = []

        # Handle images from the 'images' field
        if message.images:
            for image_data in message.images:
                image = None
                # Try to decode base64-encoded image data
                try:
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    message_content.append({"type": "image", "image": image})
                    continue  # Successfully loaded image, proceed to next
                except (base64.binascii.Error, IOError):
                    pass  # Not base64 or invalid image, try next method

                # Try to load from file path
                if os.path.exists(image_data):
                    try:
                        image = Image.open(image_data).convert("RGB")
                        message_content.append({"type": "image", "image": image})
                        continue
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Error loading image from path: {e}")

                # Try to load from URL
                try:
                    response = requests.get(image_data.strip())
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    message_content.append({"type": "image", "image": image})
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error loading image: {e}")

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
                "load_duration": 0,  # Fill this based on actual load time if tracked
                "prompt_eval_duration": 0,  # Fill if needed
                "eval_count": len(partial_response),  # Number of characters generated
                "eval_duration": int((end_time - start_time) * 1e9)  # Placeholder for eval duration
            }
            yield json.dumps(response_data) + "\n"

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
        return JSONResponse(content=response_data)

# Implement the /api/models endpoint
@app.get("/api/models")
async def list_models():
    # Return the available models
    return {"models": [model_name]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11435)
