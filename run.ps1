# PowerShell script to set up, train, and run the HR Attrition Project
Write-Host "=============================================" -ForegroundColor Green
Write-Host "Setting up HR Attrition Prediction Project..." -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

if (-not (Test-Path .venv)) {
    Write-Host "[1/5] Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[1/5] Virtual environment already exists." -ForegroundColor Yellow
}

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
. .venv\Scripts\Activate.ps1

Write-Host "[2/5] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "[3/5] Executing Data Prep & SQLite Database setup..." -ForegroundColor Yellow
python src/data_prep.py

Write-Host "[4/5] Training machine learning models & generating PDF Report..." -ForegroundColor Yellow
python src/train_model.py
python src/generate_pdf.py

Write-Host "[5/5] Running pytest suite..." -ForegroundColor Yellow
pytest tests/

Write-Host "Setup complete. Launching Streamlit web application..." -ForegroundColor Green
streamlit run dashboard/streamlit_app.py
