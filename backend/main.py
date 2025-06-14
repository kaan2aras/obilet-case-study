from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from search import combined_search

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load images data
IMAGES_PATH = os.path.join(os.path.dirname(__file__), "../data/images.json")
with open(IMAGES_PATH, "r") as f:
    images = json.load(f)

# Placeholder routers (to be implemented)
@app.get("/")
def root():
    return {"message": "oBilet Hotel Visual Search API"}

@app.post("/search")
async def search(request: Request):
    try:
        data = await request.json()
        query = data.get("query", "")
        
        if not query.strip():
            # If query is empty, return all images
            print(f"Empty query - returning all {len(images)} images")
            return {"results": images}
            
        # Call combined_search function for non-empty queries with consistent parameters
        results = combined_search(
            query=query,
            captions=images,
            embeddings_path=os.path.join(os.path.dirname(__file__), "../data/embeddings.npy"),
            top_k=10,  # Consistent with search.py
            min_match_ratio=0.5,
            feature_sim_threshold=0.85
        )
        found = [img for img in images if img["url"] in results]
        
        # Print debug information
        image_numbers = [img["url"].split("/")[-1].split(".")[0] for img in found]
        print(f"Search query: '{query}'")
        print(f"Found {len(found)} matches: {image_numbers}")
        return {"results": found}
    except Exception as e:
        print(f"Error in /search endpoint: {str(e)}")
        return {"results": [], "error": str(e)}

# TODO: Add endpoints for image_captioning, search, agent
