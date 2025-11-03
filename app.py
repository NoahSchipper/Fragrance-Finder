"""
Fragrance Selector API - FastAPI Backend
-----------------------------------------
This API provides three modes of fragrance recommendation:
1. Similar fragrances based on a selected fragrance
2. Fragrances filtered by selected notes
3. Random fragrance with optional filters

Deploy to Render with embeddings.npy and fragrances.json

Requirements: fastapi, uvicorn, numpy, scikit-learn
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import random
from typing import List, Optional
from pathlib import Path
import os
import urllib.request

# ---------------------
# Utility functions
# ---------------------
def safe_float(val, default=0.0):
    """Convert to float and make sure it is JSON serializable"""
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default

def safe_int(val, default=None):
    """Convert to int safely"""
    try:
        return int(float(val))  # Convert float to int safely
    except (TypeError, ValueError):
        return default

    
def sanitize_json(obj):
    """Recursively sanitize a dict or list for JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        # Optional: cap extremely large floats
        if abs(obj) > 1e308:
            return 0.0
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    else:
        return obj

# ---------------------
# FastAPI setup
# ---------------------
app = FastAPI(
    title="Fragrance Selector API",
    description="AI-powered fragrance recommendation system",
    version="1.0.0"
)

# CORS - adjust origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# Global variables
# ---------------------
EMBEDDINGS = None
FRAGRANCES = None

# ---------------------
# Request models
# ---------------------
class SimilarRequest(BaseModel):
    perfume_name: str
    limit: Optional[int] = 10

class NoteRequest(BaseModel):
    notes: List[str]
    limit: Optional[int] = 50

class RandomRequest(BaseModel):
    gender: Optional[str] = None
    min_rating: Optional[float] = 0.0
    year_min: Optional[int] = None
    year_max: Optional[int] = None

# ---------------------
# File download
# ---------------------
EMBEDDINGS_URL = os.getenv("EMBEDDINGS_URL", "")
FRAGRANCES_URL = os.getenv("FRAGRANCES_URL", "")

def download_file(url, filename):
    if Path(filename).exists():
        print(f"[INFO] {filename} already exists, skipping download")
        return
    if not url:
        print(f"[WARNING] No URL provided for {filename}")
        return
    print(f"[DOWNLOADING] {filename} from URL...")
    urllib.request.urlretrieve(url, filename)
    print(f"[SUCCESS] Downloaded {filename}")

# ---------------------
# Startup event
# ---------------------
@app.on_event("startup")
async def load_data():
    global EMBEDDINGS, FRAGRANCES
    print("[STARTUP] Loading data files...")

    if EMBEDDINGS_URL:
        download_file(EMBEDDINGS_URL, 'embeddings.npy')
    if FRAGRANCES_URL:
        download_file(FRAGRANCES_URL, 'fragrances.json')

    embeddings_path = Path('embeddings.npy')
    if not embeddings_path.exists():
        raise FileNotFoundError("embeddings.npy is required")
    EMBEDDINGS = np.load('embeddings.npy')
    print(f"[SUCCESS] Loaded embeddings: {EMBEDDINGS.shape}")

    fragrances_path = Path('fragrances.json')
    if not fragrances_path.exists():
        raise FileNotFoundError("fragrances.json is required")
    with open('fragrances.json', 'r', encoding='utf-8') as f:
        FRAGRANCES = json.load(f)
    
    # Sanitize Year to int
    for frag in FRAGRANCES:
        if 'Year' in frag and frag['Year'] is not None:
            try:
                frag['Year'] = int(frag['Year'])
            except (ValueError, TypeError):
                frag['Year'] = None
    print(f"[SUCCESS] Loaded {len(FRAGRANCES)} fragrances")
    print("[STARTUP] API ready!")

# ---------------------
# Root endpoint
# ---------------------
@app.get("/")
async def root():
    return {
        "status": "online",
        "api": "Fragrance Selector",
        "version": "1.0.0",
        "fragrances_loaded": len(FRAGRANCES) if FRAGRANCES else 0,
        "endpoints": {
            "similar": "/api/recommend/similar",
            "by_notes": "/api/recommend/by-notes",
            "random": "/api/recommend/random"
        }
    }

# ---------------------
# Similar fragrances
# ---------------------
@app.post("/api/recommend/similar")
async def find_similar_fragrances(request: SimilarRequest):
    perfume_name = request.perfume_name.strip()
    limit = min(request.limit, 50)
    
    if not perfume_name:
        raise HTTPException(status_code=400, detail="Perfume name is required")
    
    perfume = None
    perfume_idx = None
    for idx, frag in enumerate(FRAGRANCES):
        if frag.get('Perfume', '').lower() == perfume_name.lower():
            perfume = frag
            perfume_idx = idx
            break
    if perfume is None:
        for idx, frag in enumerate(FRAGRANCES):
            if perfume_name.lower() in frag.get('Perfume', '').lower():
                perfume = frag
                perfume_idx = idx
                break
    if perfume is None:
        raise HTTPException(status_code=404, detail=f"Perfume '{perfume_name}' not found.")
    
    query_embedding = EMBEDDINGS[perfume_idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, EMBEDDINGS)[0]
    top_indices = np.argsort(similarities)[-(limit+1):-1][::-1]

    results = []
    for idx in top_indices:
        if idx != perfume_idx:
            frag = FRAGRANCES[idx].copy()
            frag['similarity_score'] = safe_float(similarities[idx])
            results.append(frag)
    
    return sanitize_json({
        "query": perfume,
        "results": results[:limit]
    })

# ---------------------
# Find by notes
# ---------------------
@app.post("/api/recommend/by-notes")
async def find_by_notes(request: NoteRequest):
    notes = [note.strip().lower() for note in request.notes if note.strip()]
    limit = min(request.limit, 100)
    
    if not notes:
        raise HTTPException(status_code=400, detail="At least one note is required")
    
    matches = []
    for frag in FRAGRANCES:
        all_notes = ' '.join([str(frag.get('Top', '')), str(frag.get('Middle', '')), str(frag.get('Base', ''))]).lower()
        match_count = sum(1 for note in notes if note in all_notes)
        if match_count > 0:
            frag_copy = frag.copy()
            frag_copy['match_count'] = match_count
            frag_copy['match_percentage'] = round((match_count / len(notes)) * 100, 1)
            matches.append(frag_copy)
    
    matches.sort(key=lambda x: (x['match_count'], safe_float(x.get('Rating Value', 0))), reverse=True)
    
    return sanitize_json({
        "query_notes": notes,
        "total_matches": len(matches),
        "results": matches[:limit]
    })

# ---------------------
# Random fragrance
# ---------------------
@app.get("/api/recommend/random")
async def get_random_fragrance(
    gender: Optional[str] = None,
    min_rating: float = 0.0,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
):
    filtered = FRAGRANCES.copy()
    
    if gender:
        gender_lower = gender.lower()
        filtered = [f for f in filtered if gender_lower in str(f.get('Gender', '')).lower()]
    
    if min_rating > 0:
        filtered = [f for f in filtered if safe_float(f.get('Rating Value', 0)) >= min_rating]
    
    if year_min is not None:
        filtered = [f for f in filtered if safe_int(f.get('Year')) is not None and safe_int(f.get('Year')) >= year_min]
    
    if year_max is not None:
        filtered = [f for f in filtered if safe_int(f.get('Year')) is not None and safe_int(f.get('Year')) <= year_max]
    
    if not filtered:
        raise HTTPException(status_code=404, detail="No fragrances match the specified filters")
    
    result = random.choice(filtered)
    
    return sanitize_json({
        "filters_applied": {
            "gender": gender,
            "min_rating": min_rating,
            "year_range": f"{year_min or 'any'}-{year_max or 'any'}"
        },
        "total_matching": len(filtered),
        "result": result
    })

# ---------------------
# List fragrances
# ---------------------
@app.get("/api/fragrances/list")
async def list_fragrances(
    limit: int = 100,
    offset: int = 0,
    search: Optional[str] = None
):
    fragrances = FRAGRANCES
    
    if search:
        search_lower = search.lower()
        fragrances = [
            f for f in fragrances
            if search_lower in f.get('Perfume', '').lower() or
               search_lower in f.get('Brand', '').lower()
        ]
    
    total = len(fragrances)
    results = fragrances[offset:offset + limit]
    
    return sanitize_json({
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": results
    })

# ---------------------
# List all notes
# ---------------------
@app.get("/api/notes/list")
async def list_all_notes():
    notes = set()
    
    for frag in FRAGRANCES:
        for note_type in ['Top', 'Middle', 'Base']:
            note_text = frag.get(note_type, '')
            if note_text:
                individual_notes = note_text.replace(',', ' ').split()
                notes.update([n.strip().lower() for n in individual_notes if n.strip()])
    
    return sanitize_json({
        "total_notes": len(notes),
        "notes": sorted(list(notes))
    })

# ---------------------
# Health check
# ---------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "data_loaded": EMBEDDINGS is not None and FRAGRANCES is not None}

# ---------------------
# Run app
# ---------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
