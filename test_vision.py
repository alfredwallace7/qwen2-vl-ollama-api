from ollama import Client
import os
from PIL import Image

# Initialize client and console
client = Client(host='http://localhost:11435')

memory = []
previous_image = None
image_path = None

print("Enter image path help for commands or /exit to quit")

while True:

    user_input = input("> ")

    if user_input == "/exit":
        break
    elif user_input == "/clear":
        memory = []
        print("memory cleared")
        continue
    elif user_input == "/help":
        print("Commands: /exit, /clear, /help")
        continue
    elif os.path.exists(user_input.replace('"', "").replace("\\", "/")):
        if os.path.splitext(user_input.replace('"', "").replace("\\", "/"))[1].lower() in [".png", ".jpg", ".jpeg"]:
            image_path = user_input.replace('"', "").replace("\\", "/")
            print("image added")
        else:
            print("invalid image")
        continue

    # if memory is empty
    if previous_image is not image_path:
        chat_turn = {
            "role": "user",
            "content": user_input,
            "images": [image_path]
        }
        previous_image = image_path
    else:
        chat_turn = {
            "role": "user",
            "content": user_input
        }

    memory.append(chat_turn)

    # print(memory)

    stream = client.chat(
        model="Qwen/Qwen2-VL-7B-Instruct",
        messages=memory,
        stream=True,
    )

    llm_response = ""
    for chunk in stream:
        chk = chunk['message']['content']
        llm_response += chk
        print(chk, end='', flush=True)
    print("")

    memory.append({"role": "assistant", "content": llm_response})