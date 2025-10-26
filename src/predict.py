import os
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from peft import PeftModel # Import PeftModel for loading adapters

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_MODEL_CHECKPOINT = "t5-small" # We need the original base model
FINE_TUNED_ADAPTERS_PATH = "models/fine_tuned_t5_lora" # Path to LoRA adapters
# Alternative: Use a pre-trained summarization model
PRETRAINED_SUMMARIZATION_MODEL = "t5-small"  # This is already good for summarization
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 2
EARLY_STOPPING = True

def load_fine_tuned_model_lora():
    """Loads the base model and then the LoRA adapters."""
    if not os.path.exists(FINE_TUNED_ADAPTERS_PATH):
        raise FileNotFoundError(f"LoRA adapters not found at {FINE_TUNED_ADAPTERS_PATH}. Please train the model with LoRA first.")

    # 1. Load the original base model
    logger.info(f"Loading base model: {BASE_MODEL_CHECKPOINT}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_CHECKPOINT,
        torch_dtype=torch.float32  # Use float32 for CPU compatibility
    )
    
    # 2. Load the tokenizer (saved with the adapters)
    logger.info(f"Loading tokenizer from: {FINE_TUNED_ADAPTERS_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_ADAPTERS_PATH)

    # 3. Load the PEFT adapters onto the base model
    logger.info(f"Loading LoRA adapters from: {FINE_TUNED_ADAPTERS_PATH}")
    model = PeftModel.from_pretrained(base_model, FINE_TUNED_ADAPTERS_PATH)

    # 4. Merge adapters with base model (optional, but good for deployment to avoid PEFT overhead)
    # This creates a merged model that behaves like a fully fine-tuned model.
    model = model.merge_and_unload()
    logger.info("LoRA adapters merged into the base model.")

    model.eval() # Set model to evaluation mode

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.to("cuda")
        logger.info("Model moved to CUDA.")
    else:
        logger.info("CUDA not available. Model running on CPU.")
    
    return tokenizer, model

def summarize_text(text: str, tokenizer, model) -> str:
    """Generates a summary for the given text."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Add T5 summarization prefix
    text_with_prefix = f"summarize: {text}"

    inputs = tokenizer(
        [text_with_prefix],
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
            early_stopping=EARLY_STOPPING,
            do_sample=False,  # Use greedy decoding for more consistent results
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    summary_ids = outputs[0]
    logger.info(f"Generated token IDs: {summary_ids}")
    summary = tokenizer.decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    logger.info(f"Decoded summary: '{summary}'")
    return summary

def load_base_model():
    """Load the base T5 model for summarization."""
    logger.info(f"Loading base model: {BASE_MODEL_CHECKPOINT}")
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT)
    
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
        logger.info("Model moved to CUDA.")
    else:
        logger.info("CUDA not available. Model running on CPU.")
    
    return tokenizer, model

if __name__ == "__main__":
    # Use base model since LoRA training needs more data
    tokenizer, model = load_base_model()

    test_text = """
    A recent study published in Nature Communications details a breakthrough in renewable energy. Researchers at Stanford University have developed a novel organic solar cell that achieves an unprecedented 19% efficiency, a significant leap forward for flexible and transparent solar technologies. Unlike traditional silicon-based cells, these new organic cells are cheaper to produce, lighter, and can be integrated into various surfaces, including windows and wearable devices. This advancement could accelerate the adoption of solar power in urban environments and consumer electronics. Further research is ongoing to improve their long-term stability in diverse weather conditions.
    """
    
    test_text_2 = """
    The ancient city of Petra, a UNESCO World Heritage site in Jordan, is renowned for its rock-cut architecture, particularly the Treasury (Al-Khazneh) and the Monastery (Ad Deir). Established around the 4th century BC as the capital of the Nabataean kingdom, it was strategically located at the crossroads of major trade routes, facilitating its prosperity. The Nabataeans were skilled hydrologists, developing intricate water conservation systems in the arid desert. Although eventually conquered by the Roman Empire, Petra's unique blend of Hellenistic and indigenous architectural styles continues to captivate visitors, drawing millions of tourists annually who marvel at its historical significance and natural beauty.
    """

    print("--- Summarizing Test Text 1 ---")
    summary = summarize_text(test_text, tokenizer, model)
    print(f"Original Text:\n{test_text}\n")
    print(f"Summary:\n{summary}\n")

    print("--- Summarizing Test Text 2 ---")
    summary_2 = summarize_text(test_text_2, tokenizer, model)
    print(f"Original Text:\n{test_text_2}\n")
    print(f"Summary:\n{summary_2}\n")