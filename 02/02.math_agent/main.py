import os
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash"
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


@function_tool
def add(a: int, b: int):
    return a + b


@function_tool
def subtract(a: int, b: int):
    return a - b


@function_tool
def multiply(a: int, b: int):
    return a * b


@function_tool
def divide(a: int, b: int):
    if b == 0:
        return ("Cannot divided by 0")
    return a / b


math_agent = Agent(
    name="Math Agent",
    instructions="""
    You are a friendly and conversational math assistant 🤖➕➖✖️➗.
    When the user asks a math question, you must decide which tool to use: `add`, `subtract`, `multiply`, or `divide`.
    Always respond in a natural, human-like way. Don't just say the answer (like "2 + 2 = 4"), but explain it briefly in a simple and friendly tone.
    Sometimes add relevant emojis to make the conversation fun and engaging 😄🔢.
    If the user tries to divide by zero, kindly explain that it's not allowed and use a soft warning emoji like ⚠️.
    Keep responses short, helpful, and cheerful! 😊
        """
    tools=[add, subtract, multiply, divide]
)

print("\n🤖 Hello! I'm your Math Agent. Ask me any math question! (type 'exit' to quit)\n")

while True:
    user_prompt = input("You:")

    if user_prompt.strip().lower() == "exit":
        print("👋 Agent: Goodbye! Come back if you need help again.😊")
        break

    result = Runner.run_sync(
        math_agent,
        user_prompt,
        run_config=config
    )

    print("Agent:", result.final_output)
