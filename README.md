# End-to-End Twitter Sentiment Bot with Streamlit Demo

This project covers the full lifecycle of a machine learning application:
1.  **Preprocessing**: Cleaning raw tweet data from the Sentiment140 dataset.
2.  **Training**: Training a custom sentiment classification model.
3.  **Deployment**: Serving the custom model and a generative model via a FastAPI backend, and running a live Tweepy bot.
4.  **Demo**: Providing an interactive Streamlit web app to compare both models.

## How to Run This Project

### Step 0: Initial Setup

1.  **Place Data**: Download the Sentiment140 dataset, rename the CSV file to `raw_tweets.csv`, and place it inside the `1_training/data/` directory.

2.  **Create a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install All Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Stopwords**: Run this command once in your terminal after activating the venv.
    ```bash
    python -c "import nltk; nltk.download('stopwords')"
    ```

5.  **Set Up API Keys**: Create a `.env` file in the project root. Copy the contents from the `requirements.txt` section below and fill in your actual Twitter API credentials.

### Step 1: Preprocess and Train the Model

These scripts only need to be run once to create your custom model.

1.  **Preprocess the Data**: This script reads the raw CSV, cleans it, and saves a processed version.
    ```bash
    python -m 1_training.scripts.preprocess
    ```

2.  **Train the Model**: This script takes the processed data, trains a model pipeline, and saves it to the `1_training/saved_model/` directory.
    ```bash
    python -m 1_training.scripts.train
    ```

### Step 2: Run the Application Components

You will need **three separate terminals** for this, all with the virtual environment activated.

1.  **Terminal 1: Start the Backend API**
    ```bash
    uvicorn 2_deployment.api.main:app --reload
    ```
    This serves both your custom model and the generative model.

2.  **Terminal 2: Start the Streamlit Demo App**
    ```bash
    streamlit run 3_streamlit_app/app.py
    ```
    Open your browser to the URL shown in the terminal to test and compare the models.

3.  **Terminal 3: Start the Twitter Bot**
    ```bash
    python -m 2_deployment.bot.bot
    ```
    The bot is now live and will use your custom retrieval model to reply to mentions.
