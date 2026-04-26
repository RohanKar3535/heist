"""
Root-level server entry point — proxies to env/server/app.py.
Allows openenv validate to find server/app.py from the project root.
"""
import sys
import os

# Add env/ to path so all imports resolve
_ENV_DIR = os.path.join(os.path.dirname(__file__), "..", "env")
if _ENV_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_ENV_DIR))

from env.server.app import app, main  # noqa: F401
