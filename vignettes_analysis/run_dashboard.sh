#!/bin/bash

echo "🚀 Starting AI Model Evaluation Dashboard..."
echo "📦 Installing requirements..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Install requirements
pip install -r "$SCRIPT_DIR/streamlit_requirements.txt"

echo "🌐 Launching Streamlit dashboard..."
echo "📱 Dashboard will be available at: http://localhost:8501"
echo "🔧 Use Ctrl+C to stop the dashboard"

# Launch Streamlit dashboard from the project root
cd "$SCRIPT_DIR/.."
streamlit run vignettes_analysis/simple_dashboard.py --server.port 8501 --server.headless true 