# Titanic Survival Prediction: End-to-End PyTorch Pipeline

An end-to-end Deep Learning pipeline for survival prediction, featuring a custom PyTorch neural network and an interactive Streamlit dashboard for real-time inference and evaluation.

## Features
* **Full ML Pipeline:** Automated data fetching, preprocessing, training, and deployment.
* **Deep Learning:** Custom Neural Network architecture built with PyTorch.
* **Interactive Dashboard:** Live inference UI for testing new data and viewing metrics.
* **Detailed EDA:** Exploratory Data Analysis documented in a Jupyter Notebook.

## Dashboard Preview

<img width="1920" height="933" alt="1" src="https://github.com/user-attachments/assets/0ba74377-5cba-4086-adf3-327aedb23110" />



<img width="1920" height="888" alt="2" src="https://github.com/user-attachments/assets/dd4d8336-63f3-4ca9-95f0-7e9dae400a43" />



<img width="1920" height="908" alt="3" src="https://github.com/user-attachments/assets/495e07ce-c7d4-4e07-8175-3f4384a3d22e" />




## Prerequisites: Kaggle API Token
To fetch the dataset automatically from Kaggle, you need an API token:
1. Log in to your [Kaggle](https://www.kaggle.com/) account.
2. Go to **Settings** -> **API** -> **Create New Token**.
3. A `kaggle.json` file will be downloaded.
4. **Note:** The application will prompt you for your Kaggle Username and API Key upon the first run to authenticate the download.

---

## Setup & Installation

1. **Open Anaconda Prompt** (or your preferred terminal).
2. **Navigate to the project folder:**
   ```bash
   cd "path/to/YOUR_PROJECT_PATH"

3. **Install dependencies:**
   pip install -r requirements.txt

4. **How to Run**
A batch script is provided to automate the execution of the pipeline:
In the Anaconda Prompt, run:
 ```bash
(base) C:\Users\yourname> 

cd C:\Users\path\...

--->

"run app script.bat"
 ```

 ```bash
OR
   (base) C:\Users\yourname> 
   
   cd C:\Users\path\...
   
   
   --->
   
   python train.py
   
   --->
   
   streamlit run ds_app.py
  ```

5. Project Structure

   *ds_app.py:* The main Streamlit application for the dashboard and inference.
   
   
   *train.py:* Standalone script for training the PyTorch model and saving weights.
   
   
   *model_utils.py:* Core logic including the Neural Network class and data processing.
   
   *Data_Science_Home_Assignment_Segev_Ohana.ipynb:* Comprehensive EDA and data insights.
   
   *requirements.txt:* List of all necessary Python packages.
   
   *run_app_script.bat:* Quick-start batch file for Windows users.
