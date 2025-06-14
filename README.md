# oBilet Hotel Visual Search Case Study

## Overview
This project enables advanced hotel room image search based on user visual preferences, using OpenAI models for image-to-text, keyword and semantic search, and a modern web UI.

## Tech Stack
- **Backend:** Python, FastAPI, OpenAI API, scikit-learn, numpy
- **Frontend:** Next.js (React, TypeScript, TailwindCSS), Axios

## Project Structure
- `backend/` — Python backend (API, image captioning, search)
- `frontend/` — Next.js frontend (UI)
- `data/` — Stores image URLs, captions, and embeddings

## Setup Instructions

### Backend
1. `cd backend`
2. `python3 -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt` (or see below for manual install)
4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-key-here
   ```
5. Run the API:
   ```
   uvicorn main:app --reload
   ```

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev`

## Features
- Image-to-text (OpenAI Vision)
- Keyword and semantic search
- Web UI for queries and results
- (Optional) Agent-to-agent communication
- (Optional) User feedback, favorites, explanations

## Notes
- See `/data/images.json` for image URLs and generated captions.
- See `/backend/` for API and search logic.
- See `/frontend/` for the web UI. 