import os
import pandas as pd
import torch
import joblib
from dotenv import load_dotenv
from model_utils import (
    TitanicNN, 
    clean_df_and_encode, 
    prepare_data, 
    train_and_evaluate, 
    SELECTED_FEATURES,
    fetch_titanic_with_new_token
)

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # 1. Credentials Logic
    # Check if credentials exist in environment variables first
    user = os.getenv('KAGGLE_USERNAME')
    token = os.getenv('KAGGLE_KEY')
    
    # If not found in .env, fallback to manual input
    if not user:
        user = input("Please enter your Kaggle Username: ")
    else:
        print(f"Using Kaggle Username from environment: {user}")

    if not token:
        token = input("Please enter your Kaggle API Token: ")
    else:
        print("Using Kaggle API Token from environment.")
    
    # 2. Fetch data
    df_raw = fetch_titanic_with_new_token(user, token)
    
    if df_raw is not None:
        # 3. Preprocess and Prepare Data
        df_cleaned = clean_df_and_encode(df_raw)
        
        # Using the features imported from model_utils
        selected_features = SELECTED_FEATURES 
        
        X_train, X_val, y_train, y_val, scaler = prepare_data(df_cleaned, selected_features)
        
        # 4. Train Model
        print("Starting training process...")
        model = train_and_evaluate(X_train, y_train, X_val, y_val, input_dim=len(selected_features))
        
        # 5. Save Artifacts for Streamlit
        torch.save(model.state_dict(), 'model.pth')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(selected_features, 'features.joblib')
        
        print("Success! Artifacts saved: model.pth, scaler.joblib, features.joblib")
    else:
        print("Failed to fetch data. Please check your Kaggle credentials.")

if __name__ == "__main__":
    main()