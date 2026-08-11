"""FastAPI application initialization for PECS Evaluation API."""

from fastapi import FastAPI

app = FastAPI(
    title="PECS Evaluation API",
    description="Evaluation API and RAG Platform for PECS",
    version="1.0.0",
)
