"""
One-command launcher for the GreenQ-Fleet demo.

Usage:
    python run.py

Starts the FastAPI backend (which also serves the frontend) and opens your
default browser to the dashboard.
"""
import threading
import time
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def open_browser():
    time.sleep(2.0)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    print(f"\nGreenQ-Fleet starting at http://{HOST}:{PORT}  (Ctrl+C to stop)\n")
    uvicorn.run("backend.app:app", host=HOST, port=PORT, reload=False)
