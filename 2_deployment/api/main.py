import joblib
import os
import regex as re
from nltk.corpus import stopwords
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# --- App and Model Loading ---
app = FastAPI()
retrieval_pipeline = None
generative_tokenizer = None
generative_model = None

# We duplicate the preprocess_text function here to make the API self-contained.
# In a larger project, this could be part of a shared library.
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens).strip()

@app.on_event("startup")
def load_models():
    global retrieval_pipeline, generative_tokenizer, generative_model
    retrieval_model_path = os.path.join("1_training", "saved_model", "sentiment_pipeline.joblib")
    if os.path.exists(retrieval_model_path):
        retrieval_pipeline = joblib.load(retrieval_model_path)
        print("Retrieval model pipeline loaded.")
    else:
        print(f"Error: Retrieval model not found.")

    try:
        model_name = "microsoft/DialoGPT-small"
        generative_tokenizer = AutoTokenizer.from_pretrained(model_name)
        generative_model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Generative model loaded.")
    except Exception as e:
        print(f"Error loading generative model: {e}")

class TweetRequest(BaseModel):
    text: str

# --- Endpoint 1: Retrieval-Based Bot ---
@app.post("/predict-retrieval")
def predict_retrieval(request: TweetRequest):
    if not retrieval_pipeline:
        return {"error": "Retrieval model is not loaded."}
    
    cleaned_text = preprocess_text(request.text)
    prediction = retrieval_pipeline.predict([cleaned_text])[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    responses = {
        "Positive": "We're so happy to hear you're enjoying it! Thanks for sharing. 😊",
        "Negative": "We're very sorry to hear about your experience. Please DM us so we can help."
    }
    reply = responses.get(sentiment, "Thanks for your feedback!")
    return {"sentiment": sentiment, "reply": reply}

# --- Endpoint 2: Generative Bot ---
@app.post("/predict-generative")
def predict_generative(request: TweetRequest):
    if not generative_model or not generative_tokenizer:
        return {"error": "Generative model is not loaded."}

    set_seed(42)
    input_ids = generative_tokenizer.encode(request.text + generative_tokenizer.eos_token, return_tensors='pt')
    output_ids = generative_model.generate(
        input_ids, max_length=50, pad_token_id=generative_tokenizer.eos_token_id, do_sample=True, top_k=50
    )
    reply = generative_tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return {"reply": reply or "I'm not sure how to respond to that."}
