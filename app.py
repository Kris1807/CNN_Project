"""Vercel-friendly FastAPI entrypoint.

This keeps the main application in web_app.py so local development can keep using:
    uvicorn web_app:app --reload
while Vercel can auto-detect a standard FastAPI entry file.
"""

from web_app import app
