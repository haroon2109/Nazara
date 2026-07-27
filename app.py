import os
import sys
import json
import time
import torch
import gradio as gr

# Local module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.audio_service import AsyncAudioEngine, load_audio_waveform, clean_model_output
from src.tools import ToolDispatcher
from src.prompts import get_spatial_prompt, get_document_prompt, get_medication_prompt

# Engine placeholders to prevent crash if running locally without GPU
try:
    from src.model_engine import NazaraEngine
    engine = NazaraEngine()
except Exception as e:
    print(f"Warning: Could not initialize NazaraEngine. Mocking for UI testing. Error: {e}")
    engine = None

audio_service = AsyncAudioEngine()
tool_dispatcher = ToolDispatcher()

def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        return f"{allocated:.2f} / 12.0 GB (Kaggle T4 Limit)"
    return "CPU Mode (No VRAM)"

def extract_think_block(text: str):
    import re
    # Pull out the <|think|> token block so we can render it in the UI separate from speech
    match = re.search(r'<\|think\|>(.*?)</think>', text, flags=re.DOTALL)
    if match:
        return match.group(1).strip(), text.replace(match.group(0), "").strip()
    return "No reasoning generated.", text

def extract_json_tools(text: str):
    import re
    # basic hackathon-grade json extraction regex
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if match:
        try:
            # Validate JSON
            parsed = json.loads(match.group(0))
            return match.group(0)
        except:
            pass
    return None

def process_interaction(image, audio_in, mode_dropdown):
    # --- Strict Input Validation ---
    if image is None and audio_in is None:
        return None, 0, "Error: No input provided. Please provide an image or audio command.", "", "N/A"
        
    valid_modes = ["Spatial Navigation", "Document Reader", "Medication Safety Audit"]
    if mode_dropdown not in valid_modes:
        mode_dropdown = "Spatial Navigation"
    # -------------------------------
    
    start_time = time.time()
    
    vram_start = get_vram_usage()
    
    # 1. Determine prompt based on mode
    if mode_dropdown == "Spatial Navigation":
        prompt = get_spatial_prompt()
    elif mode_dropdown == "Document Reader":
        prompt = get_document_prompt()
    elif mode_dropdown == "Medication Safety Audit":
        prompt = get_medication_prompt()
    else:
        prompt = get_spatial_prompt()
        
    # 2. Extract Audio
    audio_waveform = None
    if audio_in:
        audio_waveform = load_audio_waveform(audio_in)

    # 3. Model Inference
    raw_response = ""
    if engine:
        raw_response = engine.process_frame(image_input=image, audio_input=audio_waveform, text_prompt=prompt)
    else:
        # MOCK PIPELINE FOR UI TESTING
        time.sleep(1.2) # Simulate latency
        if mode_dropdown == "Medication Safety Audit":
            raw_response = '<|think|> The pill bottle says Aspirin. Exp 2024. Extracting data via tools. </think> {"name": "verify_medication_expiry", "arguments": {"expiry_date": "2024-05"}}'
        else:
            raw_response = '<|think|> Obstacle detected at 2 meters, 12 o\'clock. </think> You have a coffee table directly in front of you at 2 meters. Please stop.'

    # 4. Parse Think Block
    think_block, remainder_text = extract_think_block(raw_response)
    
    # 5. Native Tool Dispatching
    tool_json = extract_json_tools(remainder_text)
    tool_log = "No tools triggered."
    final_spoken_text = remainder_text
    
    if tool_json:
        # Dispatch local tool
        tool_log = tool_dispatcher.dispatch(tool_json)
        final_spoken_text = "I have verified the information using system tools."
        
    # 6. Audio Generation
    audio_out_path = audio_service.generate_audio_sync(final_spoken_text)
    
    latency_ms = round((time.time() - start_time) * 1000, 2)
    vram_end = get_vram_usage()
    vram_log = f"Start: {vram_start} -> End: {vram_end}"

    return audio_out_path, latency_ms, think_block, tool_log, vram_log


# ==========================================
# UI Layout
# ==========================================
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
        # Left Panel
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Sensory Input")
            input_image = gr.Image(sources=["upload", "webcam"], type="pil", label="Live Camera Feed / Image Upload")
            
            gr.Markdown("#### Quick Presets")
            with gr.Row():
                preset_1 = gr.Button("Scan Room Obstacle", elem_classes=["preset-btn"])
                preset_2 = gr.Button("Audit Prescription Bottle", elem_classes=["preset-btn"])
                preset_3 = gr.Button("Read Banknote", elem_classes=["preset-btn"])
                
            input_audio = gr.Audio(sources=["microphone"], type="filepath", label="Microphone / Audio Query Input")
            mode_dropdown = gr.Dropdown(
                choices=["Spatial Navigation", "Medication Safety Audit", "Document Reader"],
                value="Spatial Navigation",
                label="Mode Selector"
            )
            submit_btn = gr.Button("Trigger Co-Pilot", elem_classes=["primary-btn"], size="lg")
            
        # Right Panel
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Gemma 4 Live Telemetry & Guidance")
            output_audio = gr.Audio(label="Real-Time Spoken Guidance", autoplay=True)
            latency_meter = gr.Number(label="⚡ System Latency Meter (ms)", elem_classes=["colored-badge"])
            
            with gr.Accordion("🧠 Gemma 4 Neural Engine Telemetry", open=True):
                think_log = gr.Textbox(label="Real-time Gemma 4 Internal Thinking Log (<|think|>)", lines=4, interactive=False)
                tool_log = gr.Textbox(label="Native Function Call Execution Stream", lines=4, interactive=False, elem_classes=["large-text"])
                vram_tracker = gr.Textbox(label="🔋 Live VRAM Footprint", lines=1, interactive=False)
                
    preset_1.click(lambda: "Spatial Navigation", None, mode_dropdown)
    preset_2.click(lambda: "Medication Safety Audit", None, mode_dropdown)
    preset_3.click(lambda: "Document Reader", None, mode_dropdown)
                
    submit_btn.click(
        fn=process_interaction,
        inputs=[input_image, input_audio, mode_dropdown],
        outputs=[output_audio, latency_meter, think_log, tool_log, vram_tracker]
    )

if __name__ == "__main__":
    # Queue is strictly needed here to prevent script-kiddies from OOM-ing the free endpoint
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="127.0.0.1", server_port=7863, share=True)
