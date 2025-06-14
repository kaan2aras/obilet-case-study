import json
import numpy as np
from openai import OpenAI
import os
import spacy
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

nlp = spacy.load("en_core_web_sm")

# Load captions
data_path = "../data/images.json"
with open(data_path, "r") as f:
    images = json.load(f)

# Collect all unique lemmatized tokens from captions
all_tokens = set()
for img in images:
    doc = nlp(img["caption"].lower())
    for token in doc:
        if token.is_alpha:
            all_tokens.add(token.lemma_)
all_tokens = sorted(list(all_tokens))
print(f"Found {len(all_tokens)} unique tokens.")

# Batch embed tokens (OpenAI API allows up to 2048 inputs per request)
batch_size = 100
embeddings = []
for i in range(0, len(all_tokens), batch_size):
    batch = all_tokens[i:i+batch_size]
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=batch
    )
    for j, emb in enumerate(response.data):
        embeddings.append(emb.embedding)
    print(f"Embedded tokens {i} to {i+len(batch)-1}")

# Save as a mapping {token: embedding}
token_to_emb = {token: np.array(embeddings[i]) for i, token in enumerate(all_tokens)}
np.save("../data/token_to_emb.npy", token_to_emb)
print("Saved token_to_emb.npy") 