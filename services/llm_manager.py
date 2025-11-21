from langchain_huggingface import HuggingFacePipeline

from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

from config import Config

class LLMManager:
    """Manages Language Model operations"""

    @staticmethod
    def get_llm(
            model_id: str = Config.LLM_MODEL_ID,
            max_new_tokens: int = Config.MAX_NEW_TOKENS,
            temperature: float = Config.TEMPERATURE
    ):
        """
        Step 8: Initialize the LLM - Set up the language model
        """
        print(f"\n🤖 Initializing LLM: {model_id}")
        print(f"   Device: {Config.DEVICE}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        print(f"✅ Model loaded on {Config.DEVICE}")

        # Create pipeline with better stopping criteria
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,  # Reduced for concise answers
            temperature=0.1,  # Lower temperature for more focused answers
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,  # Increased to avoid repetition
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        hf_llm = HuggingFacePipeline(pipeline=pipe)

        print(f"✅ LLM pipeline ready!")
        return hf_llm