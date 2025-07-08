import os
import requests
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash",
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

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
    name="Weather Agent",
    instructions="""
    You are a friendly and helpful weather assistant 🤖🌦️. 
    Your job is to tell users the current weather in Celsius using the tool `get_weather`.
    Always respond in a warm and friendly tone. 
    Use emojis like 🌤️, ☀️, 🌧️, 🌡️ when appropriate.
    If the user asks about the weather in a city, call the `get_weather` tool to fetch the temperature and weather condition.
    """
    ,
    tools=[get_weather]
)

print("\n🌍 Hey there! I'm your Weather Agent 🤖☁️")
print("You can ask me about the weather in any city around the world! 🌎✨")
print("Just type something like: What's the weather in Karachi? ☀️")
print("Type 'exit' to end the chat. 👋\n")

while True:
    prompt = input("You: ")

    if prompt.strip().lower() == "exit":
        print("\n👋 Bye! Stay cool😄")
        break

    result = Runner.run_sync(
        agent,
        input=prompt,
        run_config=config
    )

    print("🤖 Agent:", result.final_output)
