"""
Entry point for Hugging Face Spaces deployment.
This file exposes the FastAPI app from api/main.py
"""

from api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
