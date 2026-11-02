"""
Calaverita RAG API - Entry point
Día de Muertos calaveritas retrieval-augmented generation service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.router import router

app = FastAPI(
    title="Calaverita RAG API",
    description="Servicio RAG para calaveritas de Día de Muertos",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "calaverita-rag-api"}
