# Fragrance Finder API

An AI-powered fragrance recommendation system built with FastAPI. This API provides intelligent fragrance recommendations using cosine similarity on pre-computed embeddings, with support for filtering by notes, accords, gender, ratings, and more.

## Features

- **Similar Fragrances**: Find fragrances similar to a selected perfume using AI embeddings
- **Search by Accords**: Filter fragrances by their main aromatic accords
- **Random Discovery**: Get random fragrance recommendations with optional filters
- **Advanced Filtering**: Filter by gender, rating, release year, and more
- **Fast & Scalable**: Built on FastAPI with efficient NumPy operations

## API Endpoints

### Core Recommendations

#### Find Similar Fragrances
```http
POST /api/recommend/similar
```
Find fragrances similar to a given perfume using AI-powered similarity matching.

**Request Body:**
```json
{
  "perfume_name": "Chanel No 5",
  "limit": 10,
  "gender": "female"
}
```

#### Search by Accords
```http
POST /api/recommend/by-notes
```
Find fragrances matching specific aromatic accords.

**Request Body:**
```json
{
  "notes": ["woody"],
  "limit": 50,
  "gender": "male"
}
```

#### Random Fragrance
```http
GET /api/recommend/random?gender=male&min_rating=4.0&year_min=2000
```
Get a random fragrance with optional filters.

**Query Parameters:**
- `gender`: Filter by gender (male/female/unisex)
- `min_rating`: Minimum rating threshold
- `year_min`: Earliest release year
- `year_max`: Latest release year

### Data Endpoints

#### List Fragrances
```http
GET /api/fragrances/list?limit=100&offset=0&search=chanel
```

#### List All Accords
```http
GET /api/accords/list
```

#### List All Notes
```http
GET /api/notes/list
```

### Utility Endpoints

#### Root
```http
GET /
```
API information and available endpoints.

#### Health Check
```http
GET /health
```
Check if the API and data files are loaded correctly.

## Data Format

### Fragrance JSON Structure
Each fragrance in `fragrances.json` should have:
```json
{
  "Perfume": "Fragrance Name",
  "Brand": "Brand Name",
  "Gender": "male/female/unisex",
  "Year": 2020,
  "Rating Value": 4.5,
  "Main Accords": "woody, spicy, warm",
  "mainaccord1": "woody",
  "Top": "bergamot, lemon",
  "Middle": "lavender, rose",
  "Base": "sandalwood, musk"
}
```

### Embeddings File
- NumPy array (.npy format)
- Shape: (num_fragrances, embedding_dim)
- Each row corresponds to a fragrance in the same order as fragrances.json

## CORS Configuration

By default, CORS is configured to allow all origins for development. For production, modify the `allow_origins` parameter:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    ...
)
```

## Response Format

All endpoints return sanitized JSON with proper error handling for NaN and infinity values.

**Success Response:**
```json
{
  "query": { /* fragrance object */ },
  "results": [
    {
      "Perfume": "Similar Fragrance",
      "similarity_score": 0.95,
      ...
    }
  ]
}
```

**Error Response:**
```json
{
  "detail": "Error message"
}
```

## Error Handling

- `400`: Bad request (missing parameters)
- `404`: Resource not found (fragrance/no matches)
- `500`: Server error

## License

This project is provided as-is for fragrance recommendation purposes.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- All endpoints are tested
- Documentation is updated

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`