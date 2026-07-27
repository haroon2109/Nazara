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
custom_css = """
body, .gradio-container {
    background-color: #0B0F19 !important;
    color: #F8FAFC !important;
    font-family: 'Inter', 'Roboto', sans-serif !important;
}
.header-title {
    font-size: 2.5em;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #3B82F6, #10B981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2em;
}
.header-badge {
    text-align: center;
    font-size: 1.1em;
    color: #94A3B8;
    margin-bottom: 2em;
    padding: 8px;
    border-radius: 8px;
    background-color: #1E293B;
    border: 1px solid #334155;
    display: inline-block;
}
.badge-container {
    text-align: center;
}
.primary-btn {
    background: linear-gradient(90deg, #3B82F6, #10B981) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.5) !important;
    transition: all 0.3s ease !important;
}
.primary-btn:hover {
    box-shadow: 0 0 25px rgba(16, 185, 129, 0.7) !important;
    transform: translateY(-2px);
}
.preset-btn {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #E2E8F0 !important;
}
.preset-btn:hover {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.3) !important;
}
.large-text textarea {
    font-size: 1.2em !important;
    line-height: 1.5 !important;
}
.colored-badge input {
    color: #10B981 !important;
    font-weight: bold;
}
"""

with gr.Blocks(title="NAZARA Co-Pilot", css=custom_css, theme=gr.themes.Base()) as demo:
    gr.HTML('<div class="header-title">👁️ NAZARA — Edge-Native Spatial AI Co-Pilot</div>')
    gr.HTML('<div class="badge-container"><div class="header-badge">⚡ Powered by Gemma 4 E4B | 100% Offline Edge Execution | Sub-500ms Target</div></div>')
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Sensory Input")
            input_image = gr.Image(type="pil", label="Live Camera Feed / Visual Input (Optional)")
            
            gr.Markdown("#### Quick Presets")
            with gr.Row():
                preset_1 = gr.Button("Scan Room Obstacle", elem_classes=["preset-btn"])
                preset_2 = gr.Button("Audit Prescription Bottle", elem_classes=["preset-btn"])
                preset_3 = gr.Button("Read Banknote", elem_classes=["preset-btn"])
                
            input_text = gr.Textbox(
                lines=3, 
                placeholder="E.g., What obstacles are in front of me?", 
                label="Microphone / Audio Query Input (Text Fallback)"
            )
            mode_dropdown = gr.Dropdown(
                choices=["Spatial Navigation", "Medication Safety Audit", "Document Reader"],
                value="Spatial Navigation",
                label="Mode Selector",
                visible=False
            )
            submit_btn = gr.Button("Trigger Co-Pilot", elem_classes=["primary-btn"], size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Gemma 4 Live Telemetry & Guidance")
            output_audio = gr.Audio(label="Real-Time Spoken Guidance", interactive=False, autoplay=True)
            latency_meter = gr.Number(label="⚡ System Latency Meter (ms)", elem_classes=["colored-badge"], visible=False)
            
            with gr.Accordion("🧠 Gemma 4 Neural Engine Telemetry", open=True):
                output_text = gr.Textbox(lines=6, label="Real-time Gemma 4 Internal Thinking Log (<|think|>)", interactive=False)
                tool_log = gr.Textbox(label="Native Function Call Execution Stream (JSON)", lines=4, interactive=False, elem_classes=["large-text"], visible=False)
                vram_tracker = gr.Textbox(label="🔋 Live VRAM Footprint", lines=1, interactive=False, visible=False)
            
    preset_1.click(lambda: "What obstacles are in front of me?", None, input_text)
    preset_2.click(lambda: "Read this prescription bottle.", None, input_text)
    preset_3.click(lambda: "What is the denomination of this banknote?", None, input_text)

    submit_btn.click(
        fn=process_request,
        inputs=[input_image, input_text],
        outputs=[output_text, output_audio]
    )

if __name__ == "__main__":
    print("Launching NAZARA Interface...")
    demo.launch(server_name="127.0.0.1", server_port=7863, share=True)
