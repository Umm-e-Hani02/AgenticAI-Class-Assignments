import os
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel

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

faq_agent = Agent(
    name="FAQ Bot",
    instructions="You are a helpful FAQ bot. Help users with their problems, reply kindly."
)

questions = [
    "Who are you?",
    "What can you do?",
    "How can you help me?",
    "What is AI?",
    "How AI works?"
]

for idx, question in enumerate(questions, start=1):
    result = Runner.run_sync(
        faq_agent,
        input=question,
        run_config=config
    )

    # used ANSII Code for questions
    print(f"\033[34mQ{idx}: {question}\033[0m")
    print(f"A: {result.final_output}")
