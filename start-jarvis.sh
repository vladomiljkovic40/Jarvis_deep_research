#!/bin/bash
TUNNEL="${1:-https://jarvis.whitewolfdigital.com}"
TUNNEL="${TUNNEL%/}"

# ── Docker (needed for sandbox + Tika) ──
echo "Starting Docker..."
sudo service docker start > /dev/null 2>&1
sleep 2

# ── Tika ──
if ! docker ps --filter name=tika --filter status=running | grep -q tika; then
  echo "Starting Tika..."
  docker start tika > /dev/null 2>&1 || docker run -d --name tika -p 9998:9998 --restart unless-stopped apache/tika:latest
else
  echo "Tika already running."
fi

# ── Sandbox ──
if ! docker ps --filter name=jarvis-sandbox --filter status=running | grep -q jarvis-sandbox; then
  echo "Starting jarvis-sandbox..."
  docker start jarvis-sandbox > /dev/null 2>&1 || docker run -d --name jarvis-sandbox \
    -v ~/sandbox:/shared --restart unless-stopped ubuntu:24.04 sleep infinity
else
  echo "Sandbox already running."
fi

# ── SearXNG ──
pkill -f "searx/webapp.py" 2>/dev/null
sleep 1
echo "Starting SearXNG on :8888..."
(
  cd ~/searxng/searxng
  source venv/bin/activate
  export SEARXNG_SETTINGS_PATH=~/.config/searxng/settings.yml
  nohup python searx/webapp.py > ~/searxng.log 2>&1 &
)

# ── Pipelines ──
pkill -f "uvicorn main:app" 2>/dev/null
sleep 1
echo "Starting Pipelines on :9099..."
(
  cd ~/pipelines
  source ~/owui-venv/bin/activate
  nohup bash start.sh > ~/pipelines.log 2>&1 &
)
sleep 3

# ── Open WebUI ──
pkill -f "open-webui" 2>/dev/null
sleep 1
echo "Starting Open WebUI on :3000 (brain = $TUNNEL)..."
(
  source ~/owui-venv/bin/activate
  export ENABLE_PERSISTENT_CONFIG="True"
  export ENABLE_OLLAMA_API="True"
  export OLLAMA_BASE_URLS="https://your-tunnel-url.ngrok-free.dev"
  export ENABLE_WEB_SEARCH="True"
  export ENABLE_RAG_WEB_SEARCH="True"
  export WEB_SEARCH_ENGINE="searxng"
  export RAG_WEB_SEARCH_ENGINE="searxng"
  export SEARXNG_QUERY_URL="http://localhost:8888/search?q=<query>"
  nohup open-webui serve --port 3000 > ~/owui.log 2>&1 &
)

echo "Waiting for Open WebUI..."
until curl -s http://localhost:3000 > /dev/null 2>&1; do
  sleep 1
done

echo ""
echo "================== JARVIS UP =================="
echo "Open WebUI  : http://localhost:3000"
echo "SearXNG     : http://localhost:8888"
echo "Pipelines   : http://localhost:9099"
echo "Brain       : $TUNNEL"
echo "Tika        : http://localhost:9998"
echo "Sandbox     : jarvis-sandbox"
echo "Logs        : ~/searxng.log | ~/owui.log | ~/pipelines.log"
echo "==============================================="
echo "Tunable settings → Admin Settings."
