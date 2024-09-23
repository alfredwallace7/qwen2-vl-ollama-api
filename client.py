from ollama import Client
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule
import httpx

response = httpx.get('https://cdn.arstechnica.net/wp-content/uploads/2022/01/GettyImages-90790890-800x533.jpg')
response.raise_for_status()
image = response.content

# Initialize client and console
client = Client(host='http://localhost:11435')
console = Console()

# Helper to create section dividers
def section(title):
    """
    Prints a divider with a title for better UI structure.
    """
    console.print(Rule(title, style="bold cyan"))

# Chat Functions

def chat_basic():
    """
    Basic chat without specific formatting or streaming.
    """
    section("Chat: Basic Interaction")
    res = client.chat(
        model="Qwen/Qwen2-VL-7B-Instruct",
        messages=[{
            "role": "user",
            "content": "What does this image represent?",
            "images": [image]
        }],
        stream=False,
    )
    console.print(res)

def chat_json():
    """
    Chat with JSON formatted output.
    """
    section("Chat: JSON Response")
    res = client.chat(
        model="Qwen/Qwen2-VL-7B-Instruct",
        messages=[{
            "role": "user",
            "content": "What does this image represent? Only respond with the following JSON structure: {\"label\": \"<label>\", \"score\": <from 0.0 to 1.0>}.",
            "images": [image]
        }],
        format="json",
        stream=False,
    )
    console.print(res)

def chat_with_system():
    """
    Chat with a system message instructing the model to act as a Spanish teacher.
    """
    section("Chat: System Message (Spanish)")
    res = client.chat(
        model="Qwen/Qwen2-VL-7B-Instruct",
        messages=[{
            "role": "system",
            "content": "You are a Spanish teacher, you always respond in Spanish."
        },
        {
            "role": "user",
            "content": "What does this image represent?",
            "images": [image]
        }],
        stream=False,
    )
    console.print(res)

def chat_stream():
    """
    Chat with stream enabled for real-time response.
    """
    section("Chat: Streaming Response")
    stream = client.chat(
        model="Qwen/Qwen2-VL-7B-Instruct",
        messages=[{
            "role": "user",
            "content": "Describe this image.",
            "images": [image]
        }],
        stream=True,
    )
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print("")

# Generate Functions

def gen_basic():
    """
    Basic generation without specific formatting.
    """
    section("Generate: Basic Interaction")
    res = client.generate(
        model="Qwen/Qwen2-VL-7B-Instruct",
        prompt="Describe this image.",
        images=[image],
        stream=False,
    )
    console.print(res)

def gen_json():
    """
    Generate with JSON formatted output.
    """
    section("Generate: JSON Response")
    res = client.generate(
        model="Qwen/Qwen2-VL-7B-Instruct",
        prompt="What does this image represent? Only respond with the following JSON structure: {\"label\": \"<label>\", \"score\": <from 0.0 to 1.0>}.",
        images=[image],
        format="json",
        stream=False,
    )
    console.print(res)

def gen_with_system():
    """
    Generate with a system message to respond as a Spanish teacher.
    """
    section("Generate: System Message (Spanish)")
    res = client.generate(
        system="You are a Spanish teacher, you always respond in Spanish.",
        model="Qwen/Qwen2-VL-7B-Instruct",
        prompt="Describe this image.",
        images=[image],
        stream=False,
    )
    console.print(res)

def gen_stream():
    """
    Generate content with stream enabled for real-time response.
    """
    section("Generate: Streaming Response")
    stream = client.generate(
        model="Qwen/Qwen2-VL-7B-Instruct",
        prompt="Describe this image.",
        images=[image],
        stream=True,
    )
    for chunk in stream:
        print(chunk['response'], end='', flush=True)
    print("")

# Utility / Special Case Functions

def chat_tools():
    """
    Chat using the model with tools like getting the current date.
    """
    section("Chat: Using Tools")
    res = client.chat(
        model="Qwen/Qwen2-VL-7B-Instruct",
        messages=[{
            "role": "user",
            "content": "Count how many years from now this document was produced.",
            "images": [image]
        }],
        tools=[{
            "name": "get_date_now",
            "description": "Returns the current date and time."
        }],
        stream=False,
    )
    console.print(res)

# Main UI Loop

def main():
    """
    Main loop to handle user input and execute corresponding functions.
    Provides a clean and user-friendly interface.
    """
    while True:
        # Display menu
        show_menu()

        # Ask for user input
        user_input = Prompt.ask(
            "[bold cyan]Choose an option[/bold cyan]", 
            choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        )

        # Execute based on user input
        try:
            if user_input == "1":
                chat_basic()
            elif user_input == "2":
                chat_json()
            elif user_input == "3":
                chat_with_system()
            elif user_input == "4":
                chat_stream()
            elif user_input == "5":
                chat_tools()
            elif user_input == "6":
                gen_json()
            elif user_input == "7":
                gen_with_system()
            elif user_input == "8":
                gen_stream()
            elif user_input == "9":
                console.print(
                    Text("\nThank you for using the app! Goodbye!", style="bold green")
                )
                break

        except Exception as e:
            console.print(f"[red]An error occurred:[/red] {e}")

def show_menu():
    """
    Display the menu using a simple table for better UI.
    """
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option", justify="center", style="cyan", no_wrap=True)
    table.add_column("Description", style="bold white")

    table.add_row("1", "CHAT (Basic)")
    table.add_row("2", "CHAT (JSON)")
    table.add_row("3", "CHAT (System Message - Spanish)")
    table.add_row("4", "CHAT (Stream Enabled)")
    table.add_row("5", "CHAT (Tools - Date Calculation)")
    table.add_row("6", "GENERATE (JSON)")
    table.add_row("7", "GENERATE (System Message - Spanish)")
    table.add_row("8", "GENERATE (Stream Enabled)")
    table.add_row("9", "[red]EXIT[/red]")

    console.print(table)

if __name__ == "__main__":
    main()