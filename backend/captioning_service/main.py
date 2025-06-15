from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

class CaptionRequest(BaseModel):
    url: str

@app.post("/caption")
async def caption_image(req: CaptionRequest):
    # You can add error handling here
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this hotel room image in detail, including room type, view, amenities, and style."},
                    {"type": "image_url", "image_url": {"url": req.url}}
                ]
            }
        ],
        max_tokens=200
    )
    caption = response.choices[0].message.content.strip()
    return {"caption": caption}