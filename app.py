from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="LLM Deployment API")

# Load model and tokenizer directly
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "LLM API is running"}

@app.post("/generate")
def generate(request: PromptRequest):
    # Encode prompt, generate tokens, and decode output
    inputs = tokenizer(request.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"response": response_text}