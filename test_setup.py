import os
from dotenv import load_dotenv
from google import genai

# Load the .env file so we can read the API key
load_dotenv()

# Create a client using our Gemini API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Send a tiny test message to Gemini
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Say hello in one short sentence."
)

print(response.text)