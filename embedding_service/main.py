from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed_text(req: EmbedRequest):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=req.text
    )
    embedding = response.data[0].embedding
    return {"embedding": embedding}