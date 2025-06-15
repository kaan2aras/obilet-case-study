# oBilet Hotel Visual Search Case Study

## Overview
This project is a microservices-based hotel room search system that enables advanced image-based and feature-based hotel room filtering. It uses OpenAI models for image captioning and embedding, NLP for feature extraction, and a modern web UI for search and results.

## Architecture
The system is composed of the following microservices:

- **captioning_service/**: Generates detailed captions for hotel room images using OpenAI Vision API.
- **embedding_service/**: Provides text embedding vectors using OpenAI API.
- **search_service/**: Handles all search logic, including NLP, feature extraction, and semantic/structured filtering.
- **orchestrator_service/**: Orchestrates requests between services and provides a unified API for the frontend.
- **frontend/**: Next.js (React) web UI for users to search and view results.
- **data/**: Contains `images.json` with image URLs and captions.

## Project Structure
```
Obilet_Case_Study/
  captioning_service/
  embedding_service/
  search_service/
  orchestrator_service/
  frontend/
  data/
```

## Setup Instructions

### 1. Captioning Service
```bash
cd captioning_service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Set your OpenAI API key in a .env file:
# OPENAI_API_KEY=your-key-here
uvicorn main:app --reload --port 8001
```

### 2. Embedding Service
```bash
cd embedding_service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Set your OpenAI API key in a .env file:
# OPENAI_API_KEY=your-key-here
uvicorn main:app --reload --port 8002
```

### 3. Search Service
```bash
cd search_service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --reload --port 8003
```

### 4. Orchestrator Service
```bash
cd orchestrator_service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 5. Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000 in your browser
```

## Quickstart
1. Start all backend services (captioning, embedding, search, orchestrator) in separate terminals.
2. Start the frontend.
3. Open [http://localhost:3000](http://localhost:3000) to use the app.

## Notes
- All service ports are configurable; defaults are shown above.
- The orchestrator service loads image/caption data from `data/images.json`.
- You need valid OpenAI API keys for captioning and embedding services.
- For development, you can use `--reload` for hot-reloading FastAPI services.

## Features
- Visual and feature-based hotel room search
- Compound and custom queries (e.g., "double room with sea view and desk")
- Modern, responsive web UI
- Microservices architecture for scalability and modularity 