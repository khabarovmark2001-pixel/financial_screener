#!/bin/bash
# Start the screener. Serves on this machine and to the local network.
cd "$(dirname "$0")" || exit 1

if pgrep -f "streamlit run app.py" > /dev/null; then
  echo "Already running (pid $(pgrep -f 'streamlit run app.py' | head -1))."
else
  nohup .venv/bin/streamlit run app.py \
      --server.port 8501 --server.address 0.0.0.0 --server.headless true \
      > streamlit.log 2>&1 &
  # Wait for the health endpoint rather than a fixed sleep, so the URLs below
  # are only printed once the server can actually answer.
  for _ in $(seq 1 30); do
    curl -s --max-time 2 http://localhost:8501/_stcore/health | grep -q ok && break
    sleep 1
  done
fi

LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
echo
echo "  This machine :  http://localhost:8501"
[ -n "$LAN_IP" ] && echo "  Same Wi-Fi   :  http://$LAN_IP:8501"
echo
echo "  Log   : streamlit.log"
echo "  Stop  : ./stop_app.sh"
