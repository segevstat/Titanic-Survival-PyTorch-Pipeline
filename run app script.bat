@echo off
echo Step 1: Navigating to project directory...
cd /d C:\Users\yourpath\

echo Step 2: Starting Model Training...
python train.py

echo Step 3: Launching Streamlit App...
streamlit run ds_app.py


pause
