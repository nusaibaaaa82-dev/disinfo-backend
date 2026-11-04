from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str
    model_id: str

loaded_models = {}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        if request.model_id not in loaded_models:
            tokenizer = AutoTokenizer.from_pretrained(request.model_id)
            model = AutoModelForSequenceClassification.from_pretrained(request.model_id)
            loaded_models[request.model_id] = (tokenizer, model)
        
        tokenizer, model = loaded_models[request.model_id]
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        pred_id = torch.argmax(logits, dim=-1).item()
        label = model.config.id2label.get(pred_id, str(pred_id))
        score = float(torch.softmax(logits, dim=-1)[0][pred_id])
        
        return {"label": label, "score": score}
    except Exception as e:
        return {"error": str(e)}
