# Qwen2-VL Ollama API Emulator for models not yet supported

This project replicates Ollama API endpoints for models that are not yet supported. It uses the `Qwen2-VL-7B-Instruct` model from Hugging Face and provides endpoints for text generation and chat functionality similar to Ollama's API.

## Credits

- **[Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)**  
  Qwen2-VL is an advanced vision-language model developed by [QwenLM](https://github.com/QwenLM). It supports various vision-language tasks, including image captioning, visual question answering, and multi-modal content generation.

- **[Ollama](https://ollama.com/)**  
  Ollama provides a powerful framework for running and deploying LLMs, offering flexible APIs for various natural language processing tasks.

## Features

- Compatible with Python's **Ollama client library.**
- **Text generation** via `/api/generate`
- **Chat conversation** support via `/api/chat`
- **Image handling** (base64, file paths, or URLs)
- **Streaming** or non-streaming responses.
- Customizable prompts, generation options, and **system messages**.

## Requirements

- Python 3.8+
- PyTorch with CUDA (optional for GPU acceleration)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- FastAPI and Uvicorn for the web framework

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/qwen2vl-ollama-api.git
   cd qwen2vl-ollama-api
   ```

2. Install dependencies:

   ```bash
   pip install git+https://github.com/huggingface/transformers.git
   pip install -r requirements.txt
   ```

   Requirements include:
   - FastAPI
   - Uvicorn
   - Hugging Face Transformers
   - Torch
   - Pillow
   - Requests
   - Pydantic

3. Ensure you have the `Qwen/Qwen2-VL-7B-Instruct` model downloaded:

   ```bash
   transformers-cli download Qwen/Qwen2-VL-7B-Instruct
   ```

## Running the Server

To start the FastAPI server locally, run the following command:

```bash
python .\api.py --host=0.0.0.0 --port=11435
```

The server will be available at `http://127.0.0.1:11435`.

## API Endpoints

### `/api/generate`

Generates text based on a given prompt and optional images.

#### Request:

```json
POST /api/generate
{
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "prompt": "Explain quantum physics in simple terms.",
    "options": {
        "temperature": 0.7,
        "max_tokens": 100
    },
    "stream": true,
    "format": "json",  # Use this to get output in valid JSON format
    "images": ["base64_image_data_or_url"],
    "system_prompt": "You are a helpful assistant."
}
```

#### Response:

- **Streaming**: JSONL responses with `done: false` and a final message with `done: true`.
- **Non-streaming**: JSON object with the complete response.

### `/api/chat`

Supports multi-turn conversations with optional images.

#### Request:

```json
POST /api/chat
{
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "Tell me a joke."
        }
    ],
    "options": {
        "temperature": 0.8
    },
    "stream": true
}
```

#### Response:

Similar to `/api/generate`, supports streaming or non-streaming JSON responses.

### `/api/models`

Lists all available models endpoint for compatibility.

#### Response:

```json
{
    "models": ["Qwen/Qwen2-VL-7B-Instruct"]
}
```

## Configuration

- **Model and Device:** The model is loaded using the `Qwen/Qwen2-VL-7B-Instruct` transformer, and it automatically selects GPU if available. Around 17GB of VRAM is required for the 7B model.
- **Streaming:** You can enable or disable streaming via the `stream` parameter in the API requests.
- **Images:** Supports base64-encoded images, file paths, and URLs.
- **JSON Format:** The `format="json"` option ensures the output is formatted in valid JSON, which is useful for structured data processing.

## License

MIT License

