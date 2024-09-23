# Qwen2-VL API Emulator for Ollama Models

This project replicates Ollama API endpoints for models that are not yet supported. It uses the `Qwen2-VL-7B-Instruct` model from Hugging Face and provides endpoints for text generation and chat functionality similar to Ollama's API.

## Features

- **Text generation** via `/api/generate`
- **Chat conversation** support via `/api/chat`
- **Image handling** (base64, file paths, or URLs)
- Streaming or non-streaming responses.
- Customizable prompts, generation options, and system messages.
- Compatible with Python's Ollama library.

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
uvicorn main:app --host 0.0.0.0 --port 11435
```

The server will be available at `http://localhost:11435`.

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

Lists all available models.

#### Response:

```json
{
    "models": ["Qwen/Qwen2-VL-7B-Instruct"]
}
```

## Configuration

- **Model and Device:** The model is loaded using the `Qwen/Qwen2-VL-7B-Instruct` transformer, and it automatically selects GPU if available.
- **Streaming:** You can enable or disable streaming via the `stream` parameter in the API requests.
- **Images:** Supports base64-encoded images, file paths, and URLs.
- **JSON Format:** The `format="json"` option ensures the output is formatted in valid JSON, which is useful for structured data processing.

## License

MIT License
