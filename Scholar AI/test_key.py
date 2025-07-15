import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Embedding.create(
  input="Hello world",
  model="text-embedding-ada-002"
)

print(response)
