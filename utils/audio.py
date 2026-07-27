import os
import uuid
from gtts import gTTS

def text_to_speech(text, output_dir="temp_audio"):
    """
    Converts text to speech using gTTS and returns the path to the audio file.
    """
    if not text or not text.strip():
        return None
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"response_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join(output_dir, filename)
    
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filepath)
        return filepath
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None
