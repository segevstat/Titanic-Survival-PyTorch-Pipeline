#!/usr/bin/env python
# coding: utf-8

# In[48]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim


# In[1]:



def fetch_titanic_with_new_token(username, token, output_path="data/"):
    """
    Fetches the Titanic dataset using the new Kaggle API Token format.
    Ensures compliance with the latest Kaggle security standards.
    """
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_API_TOKEN'] = token 

    from kaggle.api.kaggle_api_extended import KaggleApi
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    api = KaggleApi()
    try:
        api.authenticate()
        print(f"Authenticated as {username}")
        
        api.competition_download_file("titanic", "train.csv", path=output_path)
        
        file_path = os.path.join(output_path, "train.csv")
        zip_path = file_path + ".zip"
        if os.path.exists(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            os.remove(zip_path)
            
        df = pd.read_csv(file_path)
        print(f"Success! Data loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None


# In[49]:


def clean_df_and_encode(df):
    
    df_cleaned= df.copy()
    
    df_cleaned['FamilySize'] = df_cleaned['SibSp'] + df_cleaned['Parch'] + 1
    
    df_cleaned['Age'] = df_cleaned['Age'].fillna(
        
        df_cleaned.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
    
    embarked_mode = df_cleaned['Embarked'].mode()[0]
        
    df_cleaned['Embarked'] = df_cleaned['Embarked'].fillna(embarked_mode)
    
    df_cleaned['Sex'] = df_cleaned['Sex'].map({'female': 0, 'male': 1})
    
    df_cleaned['Embarked'] = df_cleaned['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    return df_cleaned


# In[50]:


def prepare_data(df, SELECTED_FEATURES):
    
    X = df[SELECTED_FEATURES]
    
    y = df['Survived']

    # 2. Splitting (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scaling
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_val_scaled = scaler.transform(X_val)
    
    # 3. Convert to tensors

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    
    y_val_tensor = torch.FloatTensor(y_val.values).view(-1, 1)

    return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, scaler


# In[51]:


class TitanicNN(nn.Module):
    
    def __init__(self, input_dim):
        super(TitanicNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# In[52]:


import joblib
import os


if os.path.exists('features.joblib'):
    
    SELECTED_FEATURES = joblib.load('features.joblib')
    print("Successfully loaded features from 'features.joblib'.")
else:
    SELECTED_FEATURES = default_features
    print("'features.joblib' not found. Using default feature list.")

if not SELECTED_FEATURES:
    SELECTED_FEATURES = default_features
    print("Loaded feature list was empty. Switched to default list.")

print("--- Saved Feature List Verification ---")
print(f"Number of features: {len(SELECTED_FEATURES)}")
print(f"Feature list: {SELECTED_FEATURES}")

# 3. Final check for critical columns
if 'Sex' in SELECTED_FEATURES and 'Age' in SELECTED_FEATURES:
    print("\nFeature list appears valid and contains critical columns.")
else:
    print("\nWarning: Critical features are missing from the current list!")


# In[53]:


# 1. Load the raw data
df_raw = pd.read_csv(r'data\train.csv')

# 2. Clean and encode the data using our custom function
df_cleaned = clean_df_and_encode(df_raw)

# 3. Prepare tensors and scaling using the loaded features list
X_train, X_val, y_train, y_val, scaler = prepare_data(df_cleaned, SELECTED_FEATURES)

print("--- Training Data Verification ---")
print(f"X_train Tensor Shape: {X_train.shape}")
print(f"y_train Tensor Shape: {y_train.shape}")

# Verify scaling (Mean should be close to 0)
fare_mean = X_train[:, 3].mean().item()
print(f"Scaling Check (Mean of Fare): {fare_mean:.4f} (Expected: close to 0)")

if isinstance(X_train, torch.Tensor):
    print("\nData converted to PyTorch tensors and ready for training")


# In[54]:


def train_and_evaluate(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, epochs_num=100, input_dim=8):
    # 1. Initialize the Model 
    model = TitanicNN(input_dim=input_dim)

    # 2. Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 3. Training Loop
    for epoch in range(epochs_num):
        
        model.train()
        
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        
        loss = criterion(outputs, y_train_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Validation check every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                predictions = (val_outputs > 0.5).float()
                accuracy = (predictions == y_val_tensor).float().mean()
                print(f'Epoch [{epoch+1}/{epochs_num}], Loss: {loss.item():.4f}, Val Acc: {accuracy.item():.4f}')
    
    return model 


# In[55]:



model = train_and_evaluate(

    X_train_tensor=X_train, 

    y_train_tensor=y_train, 

    X_val_tensor=X_val, 

    y_val_tensor=y_val, 

    epochs_num=100,  

    input_dim=len(SELECTED_FEATURES)

)

