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

# Initialize FastAPI app
app = FastAPI(
    title="Fragrance Selector API",
    description="AI-powered fragrance recommendation system",
    version="1.0.0"
)

# CORS - Allow your Cloudflare Pages site to call this API
# IMPORTANT: Replace "*" with your actual domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://yourdomain.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for data (loaded once at startup)
EMBEDDINGS = None
FRAGRANCES = None

# Request/Response Models
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

import os
import urllib.request

# URLs for large files (set these as environment variables in Render)
EMBEDDINGS_URL = os.getenv("EMBEDDINGS_URL", "")
FRAGRANCES_URL = os.getenv("FRAGRANCES_URL", "")

def download_file(url, filename):
    """Download file if it doesn't exist"""
    if Path(filename).exists():
        print(f"[INFO] {filename} already exists, skipping download")
        return
    
    if not url:
        print(f"[WARNING] No URL provided for {filename}")
        return
    
    print(f"[DOWNLOADING] {filename} from URL...")
    urllib.request.urlretrieve(url, filename)
    print(f"[SUCCESS] Downloaded {filename}")

@app.on_event("startup")
async def load_data():
    """Load embeddings and fragrance data on server startup"""
    global EMBEDDINGS, FRAGRANCES
    
    print("[STARTUP] Loading data files...")
    
    # Download files if URLs are provided
    if EMBEDDINGS_URL:
        download_file(EMBEDDINGS_URL, 'embeddings.npy')
    if FRAGRANCES_URL:
        download_file(FRAGRANCES_URL, 'fragrances.json')
    
    # Load embeddings
    embeddings_path = Path('embeddings.npy')
    if not embeddings_path.exists():
        print("[ERROR] embeddings.npy not found!")
        raise FileNotFoundError("embeddings.npy is required")
    
    EMBEDDINGS = np.load('embeddings.npy', allow_pickle=True)
    print(f"[SUCCESS] Loaded embeddings: {EMBEDDINGS.shape}")
    
    # Load fragrances
    fragrances_path = Path('fragrances.json')
    if not fragrances_path.exists():
        print("[ERROR] fragrances.json not found!")
        raise FileNotFoundError("fragrances.json is required")
    
    with open('fragrances.json', 'r', encoding='utf-8') as f:
        FRAGRANCES = json.load(f)
    print(f"[SUCCESS] Loaded {len(FRAGRANCES)} fragrances")
    
    print("[STARTUP] API ready!")

@app.get("/")
async def root():
    """API health check and info"""
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

@app.post("/api/recommend/similar")
async def find_similar_fragrances(request: SimilarRequest):
    """
    Mode 1: Find fragrances similar to the selected fragrance
    
    Uses cosine similarity on AI embeddings to find the most similar fragrances
    """
    perfume_name = request.perfume_name.strip()
    limit = min(request.limit, 50)  # Cap at 50 results
    
    if not perfume_name:
        raise HTTPException(status_code=400, detail="Perfume name is required")
    
    # Find the perfume (case-insensitive)
    perfume = None
    perfume_idx = None
    
    for idx, frag in enumerate(FRAGRANCES):
        if frag.get('Perfume', '').lower() == perfume_name.lower():
            perfume = frag
            perfume_idx = idx
            break
    
    if perfume is None:
        # Try partial match
        for idx, frag in enumerate(FRAGRANCES):
            if perfume_name.lower() in frag.get('Perfume', '').lower():
                perfume = frag
                perfume_idx = idx
                break
    
    if perfume is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Perfume '{perfume_name}' not found. Please check spelling."
        )
    
    # Get embedding for this perfume
    query_embedding = EMBEDDINGS[perfume_idx].reshape(1, -1)
    
    # Calculate similarity with all fragrances
    similarities = cosine_similarity(query_embedding, EMBEDDINGS)[0]
    
    # Get top N most similar (excluding the query itself)
    top_indices = np.argsort(similarities)[-(limit+1):-1][::-1]
    
    # Build results
    results = []
    for idx in top_indices:
        if idx != perfume_idx:  # Skip the query perfume itself
            frag = FRAGRANCES[idx].copy()
            frag['similarity_score'] = float(similarities[idx])
            results.append(frag)
    
    return {
        "query": perfume,
        "results": results[:limit]
    }

@app.post("/api/recommend/by-notes")
async def find_by_notes(request: NoteRequest):
    """
    Mode 2: Find fragrances containing the selected notes
    
    Filters fragrances by top, middle, or base notes
    Returns matches ranked by number of matching notes
    """
    notes = [note.strip().lower() for note in request.notes if note.strip()]
    limit = min(request.limit, 100)  # Cap at 100 results
    
    if not notes:
        raise HTTPException(status_code=400, detail="At least one note is required")
    
    # Find matching fragrances
    matches = []
    
    for frag in FRAGRANCES:
        # Combine all notes into one searchable string
        all_notes = ' '.join([
            str(frag.get('Top', '')),
            str(frag.get('Middle', '')),
            str(frag.get('Base', ''))
        ]).lower()
        
        # Count matching notes
        match_count = sum(1 for note in notes if note in all_notes)
        
        if match_count > 0:
            frag_copy = frag.copy()
            frag_copy['match_count'] = match_count
            frag_copy['match_percentage'] = round((match_count / len(notes)) * 100, 1)
            matches.append(frag_copy)
    
    # Sort by match count (most matches first), then by rating
    matches.sort(
        key=lambda x: (x['match_count'], float(x.get('Rating Value', 0))),
        reverse=True
    )
    
    return {
        "query_notes": notes,
        "total_matches": len(matches),
        "results": matches[:limit]
    }

@app.get("/api/recommend/random")
async def get_random_fragrance(
    gender: Optional[str] = None,
    min_rating: float = 0.0,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
):
    """
    Mode 3: Get a random fragrance with optional filters
    
    Filters:
    - gender: "male", "female", "unisex"
    - min_rating: minimum rating value (0-5)
    - year_min/year_max: year range
    """
    
    # Start with all fragrances
    filtered = FRAGRANCES.copy()
    
    # Apply gender filter
    if gender:
        gender_lower = gender.lower()
        filtered = [
            f for f in filtered 
            if gender_lower in str(f.get('Gender', '')).lower()
        ]
    
    # Apply rating filter
    if min_rating > 0:
        filtered = [
            f for f in filtered 
            if float(f.get('Rating Value', 0)) >= min_rating
        ]
    
    # Apply year filters
    if year_min:
        filtered = [
            f for f in filtered 
            if f.get('Year') and float(f.get('Year', 0)) >= year_min
        ]
    
    if year_max:
        filtered = [
            f for f in filtered 
            if f.get('Year') and float(f.get('Year', 0)) <= year_max
        ]
    
    if not filtered:
        raise HTTPException(
            status_code=404, 
            detail="No fragrances match the specified filters"
        )
    
    # Select random fragrance
    result = random.choice(filtered)
    
    return {
        "filters_applied": {
            "gender": gender,
            "min_rating": min_rating,
            "year_range": f"{year_min or 'any'}-{year_max or 'any'}"
        },
        "total_matching": len(filtered),
        "result": result
    }

@app.get("/api/fragrances/list")
async def list_fragrances(
    limit: int = 100,
    offset: int = 0,
    search: Optional[str] = None
):
    """
    Utility endpoint: List all fragrances with pagination and search
    Useful for autocomplete/search features
    """
    fragrances = FRAGRANCES
    
    # Apply search filter
    if search:
        search_lower = search.lower()
        fragrances = [
            f for f in fragrances
            if search_lower in f.get('Perfume', '').lower() or
               search_lower in f.get('Brand', '').lower()
        ]
    
    # Apply pagination
    total = len(fragrances)
    results = fragrances[offset:offset + limit]
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": results
    }

@app.get("/api/notes/list")
async def list_all_notes():
    """
    Utility endpoint: Get all unique notes from the dataset
    Useful for building the note selector UI
    """
    notes = set()
    
    for frag in FRAGRANCES:
        # Extract notes from Top, Middle, Base
        for note_type in ['Top', 'Middle', 'Base']:
            note_text = frag.get(note_type, '')
            if note_text:
                # Split by common separators
                individual_notes = note_text.replace(',', ' ').split()
                notes.update([n.strip().lower() for n in individual_notes if n.strip()])
    
    return {
        "total_notes": len(notes),
        "notes": sorted(list(notes))
    }

# Health check for Render
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "data_loaded": EMBEDDINGS is not None and FRAGRANCES is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
