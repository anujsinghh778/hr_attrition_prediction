#!/bin/bash
# Bash script to set up, train, and run the HR Attrition Project
set -e

echo "============================================="
echo "Setting up HR Attrition Prediction Project..."
echo "============================================="

if [ ! -d ".venv" ]; then
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv .venv
else
    echo "[1/5] Virtual environment already exists."
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "[2/5] Installing dependencies..."
pip install -r requirements.txt

echo "[3/5] Executing Data Prep & SQLite Database setup..."
python src/data_prep.py

echo "[4/5] Training machine learning models & generating PDF Report..."
python src/train_model.py
python src/generate_pdf.py

echo "[5/5] Running pytest suite..."
pytest tests/

echo "Setup complete. Launching Streamlit web application..."
streamlit run dashboard/streamlit_app.py
