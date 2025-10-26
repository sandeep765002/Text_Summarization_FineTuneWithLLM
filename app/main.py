import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from peft import PeftModel # Import PeftModel for loading adapters
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration (can be loaded from .env or constants) ---
BASE_MODEL_CHECKPOINT = os.getenv("BASE_MODEL_CHECKPOINT", "t5-small") # New env var
LORA_ADAPTERS_PATH = os.getenv("LORA_ADAPTERS_PATH", "../models/fine_tuned_t5_lora") # New env var
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "512"))
MAX_TARGET_LENGTH = int(os.getenv("MAX_TARGET_LENGTH", "128"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))
NO_REPEAT_NGRAM_SIZE = int(os.getenv("NO_REPEAT_NGRAM_SIZE", "2"))
EARLY_STOPPING = os.getenv("EARLY_STOPPING", "True").lower() == "true"

# FastAPI app instance
app = FastAPI(
    title="LLM Text Summarizer API (LoRA Fine-tuned)",
    description="API for LoRA fine-tuned T5 model to summarize long-form text.",
    version="1.0.0",
)

# Model and tokenizer will be loaded globally on startup
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class SummarizeRequest(BaseModel):
    text: str

class SummarizeResponse(BaseModel):
    summary: str

@app.on_event("startup")
async def load_model():
    """Loads the base model and then the LoRA adapters when the FastAPI app starts."""
    global tokenizer, model
    logger.info(f"Loading base model: {BASE_MODEL_CHECKPOINT}")
    logger.info(f"Loading LoRA adapters from: {LORA_ADAPTERS_PATH}")

    if not os.path.exists(LORA_ADAPTERS_PATH):
        logger.error(f"LoRA adapters path does not exist: {LORA_ADAPTERS_PATH}")
        raise RuntimeError(f"LoRA adapters not found at {LORA_ADAPTERS_PATH}. Ensure the model is trained and available.")
    
    try:
        # 1. Load the original base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_CHECKPOINT,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
        )
        
        # 2. Load the tokenizer (saved with the adapters)
        tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTERS_PATH)

        # 3. Load the PEFT adapters onto the base model
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTERS_PATH)

        # 4. Merge adapters with base model (recommended for deployment)
        model = model.merge_and_unload()
        logger.info("LoRA adapters merged into the base model for inference.")

        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

@app.get("/health", summary="Health Check")
async def health_check():
    """Checks if the API is running and the model is loaded."""
    if model is not None and tokenizer is not None:
        return {"status": "healthy", "model_loaded": True, "device": device}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/summarize", response_model=SummarizeResponse, summary="Generate Text Summary")
async def summarize(request: SummarizeRequest):
    """
    Generates a summary for the provided text using the fine-tuned T5 model.
    """
    if model is None or tokenizer is None:
        logger.error("Model not loaded. Please check server startup logs.")
        raise HTTPException(status_code=503, detail="Model is not loaded. Service unavailable.")

    try:
        inputs = tokenizer(
            [request.text],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_TARGET_LENGTH,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                early_stopping=EARLY_STOPPING
            )

        summary_ids = outputs[0]
        summary = tokenizer.decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return SummarizeResponse(summary=summary)
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during summarization: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)