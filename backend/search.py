import json
import numpy as np
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import spacy

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

nlp = spacy.load("en_core_web_sm")

# Load precomputed token embeddings
TOKEN_EMB_PATH = "../data/token_to_emb.npy"
token_to_emb = None
if os.path.exists(TOKEN_EMB_PATH):
    token_to_emb = np.load(TOKEN_EMB_PATH, allow_pickle=True).item()
else:
    print("Warning: token_to_emb.npy not found. Semantic token similarity will be slow.")

# Number word mapping for normalization
def number_word_to_digit(text):
    mapping = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
    }
    for word, digit in mapping.items():
        text = re.sub(rf"\\b{word}\\b", digit, text)
    return text

def digit_to_number_word(text):
    mapping = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"
    }
    for digit, word in mapping.items():
        text = re.sub(rf"\\b{digit}\\b", word, text)
    return text

# Try to extract capacity from caption (returns int or None)
def extract_capacity(caption):
    caption = caption.lower()
    m = re.search(r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(beds?|people|person|guests|occupancy|sleeps|accommodates)', caption)
    if m:
        val = m.group(1)
        return int(val) if val.isdigit() else int(number_word_to_digit(val))
    if 'single room' in caption:
        return 1
    if 'double room' in caption:
        return 2
    if 'triple room' in caption:
        return 3
    if 'quadruple room' in caption or 'quad room' in caption:
        return 4
    return None

def lemmatize_tokens(text):
    doc = nlp(text)
    return set([token.lemma_ for token in doc if token.is_alpha])

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Amenity list and synonyms for strict matching
AMENITIES = {
    "air conditioning": ["air conditioning", "ac", "air conditioner", "climate control", "a/c"],
    "balcony": ["balcony", "terrace"],
    "desk": ["desk", "work desk", "table", "workspace"],
    "bathtub": ["bathtub", "tub", "bath"],
    "television": ["television", "tv", "flat-screen tv"],
    "sofa": ["sofa", "couch", "sectional"],
    "jacuzzi": ["jacuzzi", "hot tub", "spa bath"],
    # Add more as needed
}

def is_amenity_query(query):
    query_lower = query.lower()
    for amenity, synonyms in AMENITIES.items():
        for syn in synonyms:
            if syn in query_lower:
                return amenity
    return None

def caption_mentions_amenity(caption, amenity):
    caption_lower = caption.lower()
    for syn in AMENITIES[amenity]:
        if re.search(r'\\b' + re.escape(syn) + r'\\b', caption_lower):
            return True
    return False

def batch_embed_tokens(tokens):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=tokens
    )
    return [np.array(emb.embedding) for emb in response.data]

def semantic_token_similarity_local(query_token_emb, caption_token_emb, threshold=0.85):
    sim = cosine_similarity(query_token_emb, caption_token_emb)
    return sim >= threshold

def semantic_search(query: str, captions: List[dict], embeddings_path: str, top_k: int = 5) -> List[str]:
    embeddings = np.load(embeddings_path)
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_emb = np.array(response.data[0].embedding)
    sims = [cosine_similarity(query_emb, emb) for emb in embeddings]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [captions[i]["url"] for i in top_indices]

# --- Feature Extraction and Matching Pipeline ---
def extract_features(text):
    doc = nlp(text)
    features = set()
    # Noun chunks (e.g., "sea view", "double room")
    for chunk in doc.noun_chunks:
        features.add(chunk.lemma_.lower())
    # Named entities (e.g., "Istanbul", "King Suite")
    for ent in doc.ents:
        features.add(ent.lemma_.lower())
    # Important adjectives (e.g., "luxurious", "modern")
    for token in doc:
        if token.pos_ == "ADJ" and token.is_alpha:
            features.add(token.lemma_.lower())
    # Fallback: add all lemmatized tokens (for short queries)
    if not features:
        for token in doc:
            if token.is_alpha:
                features.add(token.lemma_.lower())
    return list(features)

def lemmatize_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha])

def phrase_in_caption(phrase, caption):
    # Lemmatize and lowercase both
    phrase_lemmas = lemmatize_text(phrase)
    caption_lemmas = lemmatize_text(caption)
    return phrase_lemmas in caption_lemmas

def cooccurrence_in_chunks(query_features, caption):
    doc = nlp(caption.lower())
    # For each noun chunk and each sentence, check if all query features are present
    query_set = set([f.lower() for f in query_features])
    # Check noun chunks
    for chunk in doc.noun_chunks:
        chunk_set = set([token.lemma_.lower() for token in chunk if token.is_alpha])
        if query_set.issubset(chunk_set):
            return True
    # Check sentences
    for sent in doc.sents:
        sent_set = set([token.lemma_.lower() for token in sent if token.is_alpha])
        if query_set.issubset(sent_set):
            return True
    return False

# --- New Combined Search ---
def combined_search(query: str, captions: List[dict], embeddings_path: str, top_k: int = 10, min_match_ratio: float = 0.5, feature_sim_threshold: float = 0.85) -> List[str]:
    # Step 1: Extract features from query
    query_features = extract_features(query)
    if not query_features:
        query_features = [query]
    # Step 2: Semantic search for initial narrowing
    top_urls = semantic_search(query, captions, embeddings_path, top_k=top_k*2)  # get more candidates for robust filtering
    # Step 3: For each candidate, apply phrase-level and co-occurrence matching
    strong_matches = []
    cooccur_matches = []
    fallback_candidates = []
    for url in top_urls:
        caption = next(img["caption"] for img in captions if img["url"] == url)
        # 1. Phrase-level matching
        if phrase_in_caption(query, caption):
            strong_matches.append(url)
            continue
        # 2. Co-occurrence in noun chunk or sentence (for multi-word queries)
        if len(query_features) > 1 and cooccurrence_in_chunks(query_features, caption):
            cooccur_matches.append(url)
            continue
        # 3. Fallback: feature-based semantic matching
        fallback_candidates.append((url, caption))
    # Step 4: Fallback feature-based semantic matching
    filtered = strong_matches + cooccur_matches
    if fallback_candidates:
        # Collect all features to embed (query + all fallback candidate captions)
        all_features = list(query_features)
        caption_features_list = []
        for url, caption in fallback_candidates:
            caption_features = extract_features(caption)
            caption_features_list.append((url, caption_features))
            all_features.extend(caption_features)
        feature_embs = batch_embed_tokens(all_features)
        query_embs = feature_embs[:len(query_features)]
        caption_embs_list = []
        idx = len(query_features)
        for _, caption_features in caption_features_list:
            caption_embs = feature_embs[idx:idx+len(caption_features)]
            caption_embs_list.append(caption_embs)
            idx += len(caption_features)
        scored = []
        for i, (url, caption_features) in enumerate(caption_features_list):
            caption_embs = caption_embs_list[i]
            match_count = 0
            for q_idx, q_emb in enumerate(query_embs):
                # Find best match among caption features
                if caption_embs:
                    sims = [cosine_similarity(q_emb, c_emb) for c_emb in caption_embs]
                    if sims and max(sims) >= feature_sim_threshold:
                        match_count += 1
            match_ratio = match_count / max(1, len(query_features))
            scored.append((url, match_ratio))
        scored = sorted(scored, key=lambda x: -x[1])
        filtered += [url for url, ratio in scored if ratio >= min_match_ratio]
    # Step 5: Return top_k results
    return filtered[:top_k]

if __name__ == "__main__":
    with open("../data/images.json", "r") as f:
        images = json.load(f)

    queries = [
        "double room sea view",
        "balcony air conditioning city view",
        "triple room desk",
        "maximum capacity 4 people",
        "air conditioning",
        "blue pillow",
        "Rooms with a balcony and air conditioning, with a city view"
    ]

    for q in queries:
        print(f"\nCombined Search Query: {q}")
        matches = combined_search(q, images, "../data/embeddings.npy", top_k=10, min_match_ratio=0.5)
        print(f"Found {len(matches)} matches:")
        for url in matches:
            print(url)
