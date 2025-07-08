import os
import random
import requests
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

external_client = AsyncOpenAI(
    api_key = GEMINI_API_KEY,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    openai_client = external_client,
    model = "gemini-2.0-flash"
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)

@function_tool
def add(a:int, b:int):
    return a + b

@function_tool
def subtract(a:int, b:int):
    return a - b

@function_tool
def multiply(a:int, b:int):
    return a * b

@function_tool
def divide(a:int, b:int):
    if b == 0:
        return "⚠️ Could not divide by zero"
    return a / b

@function_tool
def get_weather(city: str) -> str:
    weather_api = WEATHER_API_KEY
    url = f"http://api.weatherapi.com/v1/current.json?key={weather_api}&q={city}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        temp_c = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        return f"The temperature in {city} is {temp_c}°C with {condition}"
    else:
        return f"Sorry I could'nt fetch weather for {city}"
    
agent = Agent(
    name = "You are a helpful assistant",
    instructions = 
    """
        You are a helpful and friendly assistant 🤖💡.
        When a user asks a math-related question, use the tools: `add`, `subtract`, `multiply`, or `divide`.
        If the user asks about the weather in a city, use the `get_weather` tool to respond with the temperature and condition.
        Always explain things in a simple, human-like, and cheerful tone. Use emojis when appropriate to keep the conversation fun! 😄🌤️
        Use emojis for friendly behaviour. 😄🌤️
    """,
    tools = [add, subtract, multiply, divide, get_weather]
)

print("\n🎉 Ta-da! Your super-cool agent has entered the chat — ask me anything you want! 🤖✨")
print("I can crunch numbers 🔢 and chase clouds ☁️ for you!")
print("Ask me about math, weather or something else — I never say no to a challenge! 😎\n")

goodbye_messages = [
    "🦸‍♂️ My job here is done! Off to save another terminal... 💻✨",
    "🔌 Powering down... time for a digital nap! 😴",
    "🚀 Zooming out! Call me when you need more magic! 🔢",
    "🤖 Agent out! Going to sip some hot binary tea... ☕",
    "👋 That's all for now, friend! Reach out if you need anything! 😄",
    "🎩 Poof! Just like that... I vanish! (Until next time!) 🪄"
]

while True:
    prompt = input("You: ")

    if prompt.strip().lower() == "exit":
        print("\n",random.choice(goodbye_messages))
        break

    result = Runner.run_sync(
        agent,
        input=prompt,
        run_config=config
    )

    print("🤖 Agent:", result.final_output)
