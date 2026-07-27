import re
import time
import logging
import asyncio
import threading
import os
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing edge-tts; fallback to gTTS if unavailable
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    try:
        from gtts import gTTS
    except ImportError:
        logger.error("Neither edge-tts nor gTTS is installed.")

def clean_model_output(text: str) -> str:
    """
    Strips out residual <|think|> blocks, markdown tags, and special tokens
    to ensure clean text-to-speech audio generation.
    """
    if not text:
        return ""
        
    # Remove everything between <|think|> and </think> (including the tags)
    text = re.sub(r'<\|think\|>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove generic markdown formatting (bold, italics, etc)
    text = re.sub(r'[*_#]+', '', text)
    
    # Clean up excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_audio_waveform(filepath: str):
    """
    Loads raw audio bytes into a numpy array for Gemma 4 native processing.
    """
    import librosa
    try:
        # Gemma 4 expects 16kHz mono audio usually
        waveform, _ = librosa.load(filepath, sr=16000, mono=True)
        return waveform
    except Exception as e:
        logger.error(f"Failed to load audio waveform from {filepath}: {e}")
        return None

def measure_latency(func):
    """
    Decorator to measure execution time of functions.
    Records elapsed time from input to first byte generation.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"[Latency Benchmark] {func.__name__} completed in {elapsed:.2f} ms")
        return result
    return wrapper

class AsyncAudioEngine:
    def __init__(self, voice="en-US-JennyNeural"):
        self.voice = voice

    async def _generate_edge_tts(self, text: str, output_path: str):
        """Asynchronously stream speech using edge-tts."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)

    @measure_latency
    def generate_audio_sync(self, raw_text: str, output_path: str = "output_audio.mp3") -> str:
        """
        Cleans the input text and synchronously generates the audio file.
        Returns the path to the generated audio file for Gradio fallback.
        """
        clean_text = clean_model_output(raw_text)
        
        if not clean_text:
            logger.warning("Cleaned text is empty. Skipping TTS.")
            return None

        if EDGE_TTS_AVAILABLE:
            logger.info(f"Using edge-tts to generate audio: {output_path}")
            asyncio.run(self._generate_edge_tts(clean_text, output_path))
        else:
            logger.info(f"edge-tts unavailable, using gTTS fallback to generate: {output_path}")
            tts = gTTS(text=clean_text, lang='en', lang_check=False)
            tts.save(output_path)
            
        return output_path

    def generate_audio_async(self, raw_text: str, output_path: str = "output_audio.mp3", callback=None):
        """
        Fires the TTS generation in a background thread to prevent blocking the
        main real-time inference loop.
        """
        def _task():
            try:
                result_path = self.generate_audio_sync(raw_text, output_path)
                if callback and result_path:
                    callback(result_path)
            except Exception as e:
                logger.error(f"Async TTS task failed: {e}")

        thread = threading.Thread(target=_task, daemon=True)
        thread.start()
        return thread

if __name__ == "__main__":
    print("--- Testing AsyncAudioEngine ---")
    engine = AsyncAudioEngine()
    
    test_response = "<|think|> The pill bottle says Aspirin. Exp 2024. </think> You are holding Aspirin. **It is safe to take.**"
    
    print(f"Raw Text: {test_response}")
    cleaned = clean_model_output(test_response)
    print(f"Cleaned Text: {cleaned}")
    
    output_file = "test_speech.mp3"
    print("Generating audio...")
    path = engine.generate_audio_sync(test_response, output_path=output_file)
    
    if path and os.path.exists(path):
        print(f"Success! Audio saved to {path}")
    else:
        print("Failed to generate audio.")
