from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Set, Tuple
import numpy as np
import requests
import logging
import spacy
import re
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    logger.error("Please run: python -m spacy download en_core_web_sm")
    raise

app = FastAPI()

# Define amenity patterns with variations and required context
AMENITY_PATTERNS = {
    "tv": {
        "variations": ["tv", "television", "flat screen", "flat-screen tv", "smart tv"],
        "required_context": ["tv", "television"],
        "context_distance": 3  # Words around the term to check for context
    },
    "air conditioning": {
        "variations": ["air conditioning", "air conditioner", "ac unit", "a/c"],
        "required_context": ["air conditioning", "air conditioner", "ac", "a/c"],
        "context_distance": 3
    },
    "desk": {
        "variations": ["desk", "work desk", "study table", "writing desk"],
        "required_context": ["desk", "workstation", "study area"],
        "context_distance": 3
    },
    "balcony": {
        "variations": ["balcony", "terrace", "patio", "veranda"],
        "required_context": ["balcony", "terrace", "outdoor"],
        "context_distance": 3
    },
    "sea view": {
        "variations": ["sea view", "ocean view", "water view", "beach view", "sea views"],
        "required_context": ["sea", "ocean", "water", "beach"],
        "context_distance": 5
    },
    "city view": {
        "variations": ["city view", "urban view", "town view", "city views", "cityscape"],
        "required_context": ["city", "urban", "town", "downtown"],
        "context_distance": 5
    },
    "double room": {
        "variations": ["double room", "double bed", "double beds", "double bedroom"],
        "required_context": ["double", "two beds", "2 beds"],
        "context_distance": 3
    },
    "triple room": {
        "variations": ["triple room", "triple bed", "triple beds", "triple bedroom"],
        "required_context": ["triple", "three beds", "3 beds"],
        "context_distance": 3
    },
    "4 people": {
        "variations": ["4 people", "four people", "4 persons", "four persons", "sleeps 4"],
        "required_context": ["4", "four", "people", "persons", "guests"],
        "context_distance": 5
    }
}

# Add amenity synonym map
AMENITY_SYNONYMS = {
    "tv": ["tv", "television", "flat screen", "flat-screen tv", "smart tv"],
    "air conditioning": ["air conditioning", "air conditioner", "ac", "a/c", "climate control"],
    "desk": ["desk", "work desk", "study table", "writing desk", "workstation"],
    "balcony": ["balcony", "terrace", "patio", "veranda"],
    "bathtub": ["bathtub", "tub", "bath"],
    "sofa": ["sofa", "couch"],
    "kettle": ["kettle", "tea kettle", "electric kettle"],
    "coffee": ["coffee", "coffee maker", "coffee machine"],
    "tea": ["tea", "tea set"],
    "jacuzzi": ["jacuzzi", "hot tub"],
    "wifi": ["wifi", "wi-fi", "wireless internet"],
    "shower": ["shower", "rain shower"],
    "armchair": ["armchair", "chair"],
    "vanity": ["vanity", "mirror"],
    "table": ["table", "dining table", "side table"]
}

class ImageData(BaseModel):
    url: str
    caption: str
    embedding: List[float]

class SearchRequest(BaseModel):
    query: str
    images: List[ImageData]

def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and removing extra spaces"""
    return " ".join(text.lower().split())

def get_context_window(doc: spacy.tokens.Doc, token_idx: int, window_size: int) -> str:
    """Get text within a window around a specific token"""
    start = max(0, token_idx - window_size)
    end = min(len(doc), token_idx + window_size + 1)
    return " ".join([t.text.lower() for t in doc[start:end]])

def find_amenity_matches(text: str, caption: str) -> List[Tuple[str, float]]:
    """Find amenity matches with confidence scores"""
    text_doc = nlp(text.lower())
    caption_doc = nlp(caption.lower())
    text_normalized = normalize_text(text)
    caption_normalized = normalize_text(caption)
    
    # Track unique matches to prevent duplicates
    unique_matches = {}
    
    for amenity, pattern in AMENITY_PATTERNS.items():
        # Check if any variation appears in the query
        query_has_amenity = any(var in text_normalized for var in pattern["variations"])
        if not query_has_amenity:
            continue
        
        # Look for variations in caption
        best_match_score = 0
        for i, token in enumerate(caption_doc):
            token_context = get_context_window(caption_doc, i, pattern["context_distance"])
            
            # Check for variations in this context window
            for variation in pattern["variations"]:
                if variation in token_context:
                    # Count required context words
                    context_matches = sum(1 for ctx in pattern["required_context"] 
                                       if ctx in token_context)
                    # Calculate context score based on matches and total required
                    context_score = context_matches / len(pattern["required_context"])
                    
                    if context_score > best_match_score:
                        best_match_score = context_score
        
        # Only add if we found a good match
        if best_match_score > 0.3:  # Lowered from 0.5
            unique_matches[amenity] = best_match_score
    
    return [(k, v) for k, v in unique_matches.items()]

def extract_phrases(text: str) -> List[str]:
    """Extract meaningful phrases from text"""
    doc = nlp(text.lower())
    phrases = []
    
    # Extract noun phrases
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) > 1:  # Multi-word phrases only
            phrases.append(normalize_text(chunk.text))
    
    # Extract verb phrases with objects
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            phrase = []
            for child in token.children:
                if child.dep_ in ["dobj", "pobj"]:
                    phrase.extend([t.text.lower() for t in child.subtree])
            if phrase:
                phrases.append(" ".join(phrase))
    
    return list(set(phrases))

def calculate_match_score(query: str, caption: str) -> Tuple[float, Dict[str, Any]]:
    """Calculate match score with detailed breakdown"""
    # 1. Amenity matching
    amenity_matches = find_amenity_matches(query, caption)
    amenity_score = max([score for _, score in amenity_matches]) if amenity_matches else 0
    
    # 2. Phrase matching
    query_phrases = extract_phrases(query)
    caption_phrases = extract_phrases(caption)
    
    phrase_matches = []
    for qp in query_phrases:
        for cp in caption_phrases:
            # Check for substantial overlap (at least 50% of words match)
            qp_words = set(qp.split())
            cp_words = set(cp.split())
            overlap = len(qp_words & cp_words)
            if len(qp_words) > 0 and overlap >= 0.5 * len(qp_words):  # Lowered from 0.7
                phrase_matches.append(qp)
                break
    
    phrase_score = len(phrase_matches) / max(1, len(query_phrases)) if query_phrases else 0
    
    # 3. Calculate final score with stricter thresholds
    has_amenity_query = bool(amenity_matches)
    
    if has_amenity_query:
        # For amenity queries, require both good amenity and phrase matches
        final_score = min(0.7 * amenity_score + 0.3 * phrase_score, amenity_score)
        threshold = 0.4  # Lowered from 0.6
    else:
        # For general queries, focus more on phrase matching
        final_score = 0.4 * amenity_score + 0.6 * phrase_score
        threshold = 0.3  # Lowered from 0.5
    
    details = {
        "amenity_matches": [am[0] for am in amenity_matches],
        "amenity_score": amenity_score,
        "phrase_matches": phrase_matches,
        "phrase_score": phrase_score,
        "threshold": threshold
    }
    
    return final_score, details

def extract_capacity(text: str) -> int:
    """Extracts the maximum capacity from a caption or query. Returns None if not found."""
    # Look for explicit numbers
    match = re.search(r'(?:sleeps|capacity of|max(?:imum)? capacity of|for|accommodates|up to) (\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Look for room types
    if re.search(r'triple', text, re.IGNORECASE):
        return 3
    if re.search(r'double', text, re.IGNORECASE):
        return 2
    if re.search(r'king', text, re.IGNORECASE):
        return 2
    if re.search(r'single', text, re.IGNORECASE):
        return 1
    # Look for number of beds
    match = re.search(r'(\d+)\s+(?:single|double|king|queen)\s+beds?', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_room_type(text: str) -> str:
    text = text.lower()
    # Look for explicit room type phrases
    if re.search(r'\btriple (room|suite|bed)\b', text) or re.search(r'three\s+(single|beds?)', text):
        return "triple"
    if re.search(r'\bdouble (room|suite|bed)\b', text) or re.search(r'two\s+(single|beds?)', text):
        return "double"
    if re.search(r'\bsingle (room|suite|bed)\b', text) or re.search(r'one\s+(single|bed)', text):
        return "single"
    if re.search(r'\bking (room|suite|bed)\b', text):
        return "king"
    if re.search(r'\bsuite\b', text):
        return "suite"
    if re.search(r'\bdeluxe\b', text):
        return "deluxe"
    if re.search(r'\bstandard\b', text):
        return "standard"
    # Capacity-based mapping
    match = re.search(r'(sleeps|for|accommodates|up to)\s*(\d+)', text)
    if match:
        n = int(match.group(2))
        if n == 1:
            return "single"
        if n == 2:
            return "double"
        if n == 3:
            return "triple"
    # Look for two beds/three beds
    if re.search(r'\btwo beds?\b', text):
        return "double"
    if re.search(r'\bthree beds?\b', text):
        return "triple"
    return None

def extract_view(text: str) -> str:
    text = text.lower()
    # Look for sea view context
    if re.search(r'(sea|ocean|beach) view', text):
        return "sea view"
    if re.search(r'view of the (sea|ocean|beach)', text):
        return "sea view"
    if re.search(r'balcony.*(sea|ocean|beach)', text):
        return "sea view"
    if re.search(r'window.*(sea|ocean|beach)', text):
        return "sea view"
    return None

def normalize_amenity(term: str) -> str:
    for base, synonyms in AMENITY_SYNONYMS.items():
        for s in synonyms:
            if s in term:
                return base
    return term

def extract_amenities(text: str) -> set:
    amenities = set()
    text_lower = text.lower()
    for base, synonyms in AMENITY_SYNONYMS.items():
        for s in synonyms:
            if s in text_lower:
                amenities.add(base)
    return amenities

def parse_query_features(query: str):
    # Extract capacity constraint
    max_capacity = None
    match = re.search(r'max(?:imum)? capacity of (\d+)', query, re.IGNORECASE)
    if match:
        max_capacity = int(match.group(1))
    # Extract room type
    room_type = extract_room_type(query)
    # Extract view
    view = extract_view(query)
    # Extract amenities (normalized)
    amenities = extract_amenities(query)
    return max_capacity, room_type, view, amenities

def matches_structured(room_caption: str, query: str) -> bool:
    max_capacity, room_type, view, amenities = parse_query_features(query)
    cap = extract_capacity(room_caption)
    rtype = extract_room_type(room_caption)
    rview = extract_view(room_caption)
    r_amenities = extract_amenities(room_caption)
    # Capacity filter
    if max_capacity is not None and (cap is None or cap > max_capacity):
        return False
    # Room type filter (strict: if query asks for a type, require a match)
    if room_type:
        if not rtype or rtype != room_type:
            return False
    # View filter (strict: if query asks for a view, require a match)
    if view:
        if not rview or rview != view:
            return False
    # Amenities filter (AND logic, all must be present)
    if amenities and not amenities.issubset(r_amenities):
        return False
    return True

@app.post("/search")
async def search_images(req: SearchRequest):
    logger.info(f"Received search request with query: {req.query}")
    logger.info(f"Total images in database: {len(req.images)}")
    if not req.query.strip():
        logger.info("Empty query received, returning no results")
        return {"results": [], "total_results": 0, "total_images": len(req.images), "query": req.query}
    logger.info("Getting query embedding from embedding service...")
    response = requests.post(
        "http://localhost:8002/embed",
        json={"text": req.query}
    )
    query_emb = np.array(response.json()["embedding"])
    logger.info("Successfully received query embedding")
    results = []
    # First, try structured matching
    for img in req.images:
        if matches_structured(img.caption, req.query):
            # If structured match, boost score
            semantic_score = float(np.dot(query_emb, np.array(img.embedding)) / 
                                (np.linalg.norm(query_emb) * np.linalg.norm(img.embedding)))
            results.append({
                "url": img.url,
                "score": 1.0,
                "semantic_score": semantic_score,
                "details": {"structured": True}
            })
    # If no structured matches, fallback to previous logic
    if not results:
        for img in req.images:
            match_score, match_details = calculate_match_score(req.query, img.caption)
            semantic_score = float(np.dot(query_emb, np.array(img.embedding)) / 
                                (np.linalg.norm(query_emb) * np.linalg.norm(img.embedding)))
            final_score = 0.8 * match_score + 0.2 * semantic_score
            if final_score > match_details["threshold"] and semantic_score > 0.5:
                results.append({
                    "url": img.url,
                    "score": final_score,
                    "match_score": match_score,
                    "semantic_score": semantic_score,
                    "details": match_details
                })
    results.sort(key=lambda x: -x["score"])
    logger.info(f"Found {len(results)} relevant matches:")
    if results:
        logger.info("Top matches with scores:")
        for r in results[:3]:
            logger.info(f"- {r['url'].split('/')[-1]}: score={r['score']:.3f}")
    return {
        "results": [{"url": r["url"]} for r in results],
        "total_results": len(results),
        "total_images": len(req.images),
        "query": req.query
    }