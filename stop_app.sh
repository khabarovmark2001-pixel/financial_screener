#!/bin/bash
# Stop the screener.
if pgrep -f "streamlit run app.py" > /dev/null; then
  pkill -f "streamlit run app.py"
  sleep 1
  echo "Stopped."
else
  echo "Not running."
fi
