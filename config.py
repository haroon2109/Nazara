import torch

# Model configuration
MODEL_ID = "google/gemma-4-e4b-it"
FALLBACK_MODEL_ID = "google/gemma-4-12b-it"

# Generation limits
MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKENS = 4096

# Device configuration
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_MAP = "auto"

# Quantization
# 4-bit BitsAndBytes configuration for Kaggle GPU memory optimization
QUANTIZATION = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.bfloat16
}

# Performance
LATENCY_TARGET_MS = 1200

# Audio configuration
AUDIO_SAMPLE_RATE = 16000

# Fallback Paths
FALLBACK_AUDIO_PATH = "utils/sample_audio.wav"
FALLBACK_IMAGE_PATH = "utils/sample_image.jpg"

if __name__ == "__main__":
    print("--- NAZARA Config Initialization ---")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"Allocated VRAM: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
    else:
        print("Running on CPU. Expect higher latency.")
    print("------------------------------------")
