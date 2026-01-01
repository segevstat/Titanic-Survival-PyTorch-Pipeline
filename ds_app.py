import streamlit as st
import pandas as pd
import torch
import joblib
import os
import plotly.express as px 
from dotenv import load_dotenv
from model_utils import TitanicNN, clean_df_and_encode

# Load environment variables from .env file
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

st.title("Titanic Survival Prediction Interface")
st.write("This app uses a trained PyTorch model to predict passenger survival.")

# --- Sidebar: Kaggle Authentication ---
st.sidebar.header("Kaggle Authentication")

# Fetch defaults from environment variables
default_user = os.getenv('KAGGLE_USERNAME', '')
default_key = os.getenv('KAGGLE_KEY', '')

kaggle_user = st.sidebar.text_input("Kaggle Username", value=default_user)
kaggle_key = st.sidebar.text_input("Kaggle API Token", value=default_key, type="password")

if st.sidebar.button("Fetch Data from Kaggle"):
    if kaggle_user and kaggle_key:
        os.environ['KAGGLE_USERNAME'] = kaggle_user
        os.environ['KAGGLE_KEY'] = kaggle_key
        st.sidebar.success("Credentials set!")
    else:
        st.sidebar.error("Please enter both credentials")

# --- Main Logic: Load Model Artifacts ---
@st.cache_resource
def load_assets():
    try:
        features = joblib.load('features.joblib')
        scaler = joblib.load('scaler.joblib')
        model = TitanicNN(input_dim=len(features))
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model assets: {e}. Did you run train.py first?")
        return None, None, None

model, scaler, selected_features = load_assets()

# --- File Uploader ---
st.header("Upload Dataset for Inference")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None and model is not None:
    try:
        # Load the uploaded file
        df_input = pd.read_csv(uploaded_file)
        
        # Check for essential raw columns required for preprocessing
        # This prevents errors caused by missing 'FamilySize' which is created later
        required_raw_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        missing_raw = [col for col in required_raw_cols if col not in df_input.columns]
        
        if missing_raw:
            st.error(f"Error: The uploaded file is missing essential Titanic columns: {', '.join(missing_raw)}")
            st.info("Please make sure you are uploading a Titanic-formatted CSV (e.g., train.csv).")
        else:
            st.write("### Raw Uploaded Data", df_input.head())
            
            if st.button("Run Prediction"):
                with st.spinner("Processing..."):
                    # Preprocess data (this generates FamilySize and encodes variables)
                    df_cleaned = clean_df_and_encode(df_input)
                    
                    # Extract the features the model was trained on
                    X_raw = df_cleaned[selected_features]
                    X_scaled = scaler.transform(X_raw)
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                    
                    # Run Model Inference
                    with torch.no_grad():
                        outputs = model(X_tensor)
                        predictions = (outputs > 0.5).float()
                    
                    # Map results to dataframe
                    df_input['Survival_Probability'] = outputs.numpy().flatten()
                    df_input['Prediction'] = predictions.numpy().astype(int).flatten()
                    df_input['Prediction_Label'] = df_input['Prediction'].map({1: "Survived", 0: "Perished"})
                    
                    st.success("Predictions Complete!")
                    st.write("### Results", df_input[['PassengerId', 'Name', 'Survival_Probability', 'Prediction_Label']])
                    
                    # --- Visualization Section ---
                    st.write("### Prediction Distribution")
                    
                    counts = df_input['Prediction_Label'].value_counts().reset_index()
                    counts.columns = ['Status', 'Count']
                    total = counts['Count'].sum()
                    counts['Percentage'] = (counts['Count'] / total * 100).round(1)
                
                    fig = px.bar(
                        counts, 
                        x='Status', 
                        y='Count',
                        text=counts.apply(lambda r: f"{r['Count']} ({r['Percentage']}%)", axis=1),
                        color='Status',
                        color_discrete_map={'Perished': '#EF553B', 'Survived': '#00CC96'},
                        category_orders={"Status": ["Survived", "Perished"]}
                    )

                    fig.update_traces(textposition='outside')
                    fig.update_layout(
                        yaxis_title="Number of Passengers",
                        xaxis_title="Prediction Outcome",
                        showlegend=False,
                        bargap=0.4  
                    )

                    # Layout centering with 20% larger chart (width=540)
                    col1, col2, col3 = st.columns([0.8, 2, 0.8]) 
                    with col2:
                        st.plotly_chart(fig, use_container_width=False, width=540)

    except Exception as e:
        st.error("Critical Error: Could not process the file. Please ensure it is a valid CSV file.")
        st.info("Technical details: " + str(e))