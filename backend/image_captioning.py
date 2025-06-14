import json
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_caption(image_url: str) -> str:
    """
    Uses OpenAI Vision API to generate a detailed caption for the given image URL.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this hotel room image in detail, including room type, view, amenities, and style."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def caption_all_images(json_path: str):
    with open(json_path, "r") as f:
        images = json.load(f)
    for img in images:
        if not img["caption"]:
            print(f"Generating caption for {img['url']}...")
            try:
                img["caption"] = generate_caption(img["url"])
            except Exception as e:
                print(f"Failed to generate caption for {img['url']}: {e}")
    with open(json_path, "w") as f:
        json.dump(images, f, indent=2)
    print("All captions generated and saved.")

def generate_caption_embeddings(json_path: str, out_path: str):
    with open(json_path, "r") as f:
        images = json.load(f)
    captions = [img["caption"] for img in images]
    print("Generating embeddings for all captions...")
    embeddings = []
    for caption in captions:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=caption
        )
        embeddings.append(response.data[0].embedding)
    np.save(out_path, np.array(embeddings))
    print(f"Saved embeddings to {out_path}")

if __name__ == "__main__":
    # Uncomment to generate captions:
    # caption_all_images("../data/images.json")
    # Generate and save embeddings for all captions:
    generate_caption_embeddings("../data/images.json", "../data/embeddings.npy")
