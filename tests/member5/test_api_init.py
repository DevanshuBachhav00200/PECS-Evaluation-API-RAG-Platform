"""Unit tests for API-001 FastAPI application initialization."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import app as app_from_init
from src.api.main import app as app_from_main


def test_fastapi_app_instance():
    """Verify that app is an instance of FastAPI."""
    assert isinstance(app_from_main, FastAPI)
    assert app_from_main is app_from_init


def test_fastapi_app_metadata():
    """Verify app title and configuration."""
    assert app_from_main.title == "PECS Evaluation API"


def test_fastapi_test_client():
    """Verify that TestClient can interact with the app instance."""
    client = TestClient(app_from_main)
    # App is initialized and responds (e.g. 404 for non-existent endpoint before endpoints are added)
    response = client.get("/")
    assert response.status_code == 404
