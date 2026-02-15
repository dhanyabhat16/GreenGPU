"""
GreenGPU Configuration
Environment variables and API keys. Copy to .env or set in shell.
"""
import os

# Gemini API for AI explanations (optional)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCXFgj_2fjOKsXtz_uAywK2GXLPhBC4vgc")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
