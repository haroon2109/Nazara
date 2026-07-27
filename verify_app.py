from gradio_client import Client, handle_file
import time
import os
import sys
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

def run_verification():
    print("Connecting to local NAZARA Gradio server...")
    
    # Wait for the server to be ready
    client = None
    for i in range(10):
        try:
            client = Client("http://127.0.0.1:7863/")
            break
        except Exception:
            time.sleep(2)
            
    if not client:
        print("Failed to connect to Gradio server.")
        return
        
    print("Successfully connected. Creating mock Pill Bottle image...")
    # Create a dummy image
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img.save('dummy_pill.jpg')
    
    # Create dummy audio
    with open('dummy_audio.wav', 'wb') as f:
        f.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
    
    print("Sending verification query to NAZARA (Medication Safety mode)...")
    try:
        result = client.predict(
            image=handle_file('dummy_pill.jpg'),
            audio_in=handle_file('dummy_audio.wav'),
            mode_dropdown="Medication Safety",
            api_name="/process_interaction"
        )
        print("\n--- Gradio API Response ---")
        print(f"Audio Out: {result[0]}")
        print(f"Text Response: {result[1]}")
        print(f"System Log:\n{result[2]}")
        print(f"Latency (ms): {result[3]}")
        print(f"Think Log:\n{result[4]}")
        
        # Assertions based on our MOCK_NAZARA output
        assert "Aspirin" in result[2], "Medication verification tool was not logged properly."
        assert "User is holding a pill bottle" in result[4], "The <|think|> block was not extracted."
        
        print("\n✅ Verification Successful: Native function calling fired and <|think|> token displayed in UI.")
        
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    run_verification()
