from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

from config import Config


class LLMManager:
    """Manages Language Model operations with device-aware optimization"""

    @staticmethod
    def get_llm(
            model_id: str = Config.LLM_MODEL_ID,
            max_new_tokens: int = Config.MAX_NEW_TOKENS,
            temperature: float = Config.TEMPERATURE
    ):
        """
        Step 8: Initialize the LLM - Set up the language model with device-specific optimization

        Automatically selects appropriate model loading strategy based on device:
        - GPU: Uses FP16 for speed
        - CPU: Uses 8-bit quantization for efficiency
        """
        print(f"\n🤖 Initializing LLM: {model_id}")
        print(f"   Device: {Config.DEVICE.upper()}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Device-specific model loading
        if Config.DEVICE == "cuda":
            print(f"   Loading model with FP16 precision for GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print(f"✅ Model loaded on GPU with FP16")

        else:  # CPU mode
            print(f"   Loading model with 8-bit quantization for CPU...")
            try:
                # Try 8-bit quantization first (requires bitsandbytes)
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                print(f"✅ Model loaded on CPU with 8-bit quantization")

            except Exception as e:
                print(f"⚠️  8-bit quantization failed, loading in FP32 (slower): {str(e)}")
                # Fallback to standard loading
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                print(f"✅ Model loaded on CPU with FP32 (may be slow)")

        # Create pipeline with device-optimized parameters
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100 if Config.DEVICE == "cuda" else 80,  # Slightly less for CPU
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        hf_llm = HuggingFacePipeline(pipeline=pipe)

        print(f"✅ LLM pipeline ready!")

        # Performance tip for CPU users
        if Config.DEVICE == "cpu":
            print(
                f"💡 CPU Performance Tip: First query may be slow (~10-15s), subsequent queries will be faster (~5-8s)")

        return hf_llm