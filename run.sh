#!/usr/bin/env bash
# AI Lab Studio – launch script
cd "$(dirname "$0")"
source venv/bin/activate
# Find a free port starting from 8501
PORT=8501
while lsof -iTCP:$PORT -sTCP:LISTEN > /dev/null 2>&1; do
  PORT=$((PORT+1))
done
echo "Starting Streamlit on port $PORT"
streamlit run app.py --server.port $PORT --server.headless true
