from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize FastAPI
app = FastAPI()

# Load the tokenizer and model from Hugging Face Hub
try:
    HF_MODEL_ID = "Manoj2309/my-bert-model"  # Replace with your actual model ID

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
except Exception as e:
    raise RuntimeError(f"Error loading model or tokenizer: {e}")

# Define input structure
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input_text: InputText):
    try:
        # Tokenize input
        inputs = tokenizer(
            input_text.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        return {
            "predicted_class": predicted_class,
            "confidence": probabilities[0].tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Enable CORS (important for Vercel frontend)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://front-end-gamma-flax.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
