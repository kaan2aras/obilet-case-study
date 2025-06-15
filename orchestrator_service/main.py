from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load images (with captions) at startup
with open("../data/images.json", "r") as f:
    images = json.load(f)

class SearchRequest(BaseModel):
    query: str

@app.post("/search")
async def search(req: SearchRequest):
    # 1. Get embeddings for all image captions
    images_with_embeddings = []
    for img in images:
        emb_resp = requests.post(
            "http://localhost:8002/embed",
            json={"text": img["caption"]}
        )
        embedding = emb_resp.json()["embedding"]
        images_with_embeddings.append({
            "url": img["url"],
            "caption": img["caption"],
            "embedding": embedding
        })

    # 2. Call the Search Agent
    search_resp = requests.post(
        "http://localhost:8003/search",
        json={
            "query": req.query,
            "images": images_with_embeddings
        }
    )
    results = search_resp.json()["results"]
    return {"results": results}