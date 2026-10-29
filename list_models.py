import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
for m in sorted(client.models.list().data, key=lambda m: m.id):
    print(m.id)