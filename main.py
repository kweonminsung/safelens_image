"""
SafeLens Image Privacy Sanitization Service

Run this script to start the FastAPI server.
"""

import uvicorn


def main():
    """Start the API server."""
    print("Starting SafeLens Image Privacy Sanitization API...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info",
    )


if __name__ == "__main__":
    main()
