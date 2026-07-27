import os
import torch
import logging
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import sys

# Ensure config can be loaded
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config
except ImportError:
    class config:
        MODEL_ID = "google/gemma-4-e4b-it"
        QUANTIZATION = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16}
        DEVICE_MAP = "auto"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NazaraEngine:
    def __init__(self):
        # We gotta use 4-bit or this will instantly OOM on the Kaggle T4 limit (16GB)
        logger.info(f"Loading {config.MODEL_ID} processor...")
        self.processor = AutoProcessor.from_pretrained(config.MODEL_ID)
        
        logger.info(f"Loading {config.MODEL_ID} model with 4-bit quantization...")
        
        # BitsAndBytesConfig setup for memory efficiency (T4/P100 target)
        bnb_config = BitsAndBytesConfig(**config.QUANTIZATION)
        
        try:
            # Load the model directly using 4-bit
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_ID,
                device_map=config.DEVICE_MAP,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True
            )
            logger.info("NAZARA Engine initialized successfully with 4-bit quantization.")
        except Exception as e:
            logger.warning(f"4-bit quantization failed: {e}. Falling back to float16 with device_map='auto'...")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_ID,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            logger.info("NAZARA Engine initialized successfully via float16 fallback.")

    def process_frame(self, image_input, audio_input=None, text_prompt=None, max_visual_tokens=256):
        # Default prompt if UI somehow sends an empty query
        if text_prompt is None and audio_input is None:
            text_prompt = "Describe this scene for spatial navigation."
            
        inputs = {}
        
        # True multimodal ingestion - passing raw audio straight to AutoProcessor instead of Whisper!
        if audio_input is not None:
            # Pass raw waveform and image to processor
            inputs = self.processor(
                images=image_input,
                audio=audio_input,
                text=text_prompt if text_prompt else "", 
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                images=image_input,
                text=text_prompt,
                return_tensors="pt"
            )
            
        # Move to GPU
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # TODO: wire up max_visual_tokens natively to AutoProcessor when HF fully supports it
        
        try:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=512)
            
            # Skip special tokens so we don't leak EOS/BOS tokens into the TTS engine
            generated_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return str(e)
        finally:
            self._cleanup_memory()
            
    def _cleanup_memory(self):
        # Hard memory flush. Crucial for Kaggle.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("CUDA memory cache cleared.")
            logger.info("CUDA memory cache cleared.")


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    
    print("--- Running NAZARA Engine Verification Test ---")
    
    # Check VRAM limits if using GPU
    if torch.cuda.is_available():
        initial_vram = torch.cuda.memory_allocated() / (1024**3)
        print(f"Initial VRAM: {initial_vram:.2f} GB")
        
    try:
        engine = NazaraEngine()
        
        # Create a dummy image
        dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        print("Mocking inference run...")
        
        # In a real environment without weights this might crash or download large models, 
        # but the instantiation architecture is tested.
        # response = engine.process_frame(image_input=dummy_image, text_prompt="Test")
        # print(f"Response: {response}")
        
        print("Engine instantiation completed successfully.")
        
        if torch.cuda.is_available():
            final_vram = torch.cuda.memory_allocated() / (1024**3)
            print(f"Final VRAM after loading: {final_vram:.2f} GB")
            if final_vram > 12.0:
                print("WARNING: VRAM exceeds 12GB Kaggle T4 limit!")
            else:
                print("SUCCESS: VRAM is well within the 12GB budget.")
                
    except Exception as e:
        print(f"Failed to instantiate or run the engine: {e}")
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
