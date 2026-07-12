#!/bin/bash
pkill -f "searx/webapp.py" 2>/dev/null && echo "SearXNG stopped." || echo "SearXNG not running."
pkill -f "uvicorn main:app" 2>/dev/null && echo "Pipelines stopped." || echo "Pipelines not running."
pkill -f "open-webui" 2>/dev/null && echo "Open WebUI stopped." || echo "Open WebUI not running."

echo "Stopping Docker containers..."
docker stop tika 2>/dev/null && echo "  Tika stopped." || echo "  Tika not running."
docker stop jarvis-sandbox 2>/dev/null && echo "  Sandbox stopped." || echo "  Sandbox not running."

sleep 1
echo "JARVIS stopped."
