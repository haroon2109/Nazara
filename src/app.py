import gradio as gr
import os
import sys

# Add parent directory to path to ensure modules resolve correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_engine import Gemma4Engine
from utils.audio import text_to_speech

print("Initializing NAZARA Engine...")
engine = Gemma4Engine()

def process_request(image, text):
    """
    Handler for the Gradio interface that triggers text generation and audio synthesis.
    """
    if image is None and not text.strip():
        return "Please provide an image or text prompt.", None
        
    # Default text prompt if only an image is provided
    if not text.strip() and image is not None:
        text = "Please describe what you see in this image."
        
    try:
        # Run multimodal inference
        response_text = engine.process_multimodal_input(image_input=image, text_prompt=text)
        
        # Generate Text-to-Speech response
        audio_path = text_to_speech(response_text)
        
        return response_text, audio_path
        
    except Exception as e:
        return f"An error occurred: {str(e)}", None

# Construct Gradio Interface
with gr.Blocks(title="NAZARA Co-Pilot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 👁️ NAZARA: Edge-Native Assistive Co-Pilot
        **Powered by Gemma 4 Multimodal**
        Upload an image and ask questions, or just provide a text prompt to chat. 
        NAZARA will respond with both text and voice.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Visual Input (Optional)")
            input_text = gr.Textbox(
                lines=3, 
                placeholder="E.g., What obstacles are in front of me?", 
                label="Instruction / Prompt"
            )
            submit_btn = gr.Button("Submit to NAZARA", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(lines=6, label="Text Response", interactive=False)
            output_audio = gr.Audio(label="Voice Output", interactive=False, autoplay=True)
            
    submit_btn.click(
        fn=process_request,
        inputs=[input_image, input_text],
        outputs=[output_text, output_audio]
    )

if __name__ == "__main__":
    print("Launching NAZARA Interface...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
