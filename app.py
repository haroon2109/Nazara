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
        
    valid_modes = ["Spatial Navigation", "Document Audit", "Medication Safety"]
    if mode_dropdown not in valid_modes:
        mode_dropdown = "Spatial Navigation"
    # -------------------------------
    
    start_time = time.time()
    
    vram_start = get_vram_usage()
    
    # 1. Determine prompt based on mode
    if mode_dropdown == "Spatial Navigation":
        prompt = get_spatial_prompt()
    elif mode_dropdown == "Document Audit":
        prompt = get_document_prompt()
    elif mode_dropdown == "Medication Safety":
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
        if mode_dropdown == "Medication Safety":
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
with gr.Blocks(title="NAZARA Co-Pilot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ NAZARA: Edge-Native Assistive Co-Pilot")
    gr.Markdown("Complete Gemma 4 Pipeline: Real-Time Multimodal Inference + Native Reasoning & Tool Execution.")
    
    with gr.Row():
        # Left Panel
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Feed")
            input_image = gr.Image(sources=["upload", "webcam"], type="pil", label="Visual Input")
            input_audio = gr.Audio(sources=["microphone"], type="filepath", label="Voice Command (Native Gemma 4 Input)")
            mode_dropdown = gr.Dropdown(
                choices=["Spatial Navigation", "Document Audit", "Medication Safety"],
                value="Spatial Navigation",
                label="Query Mode"
            )
            submit_btn = gr.Button("Trigger Co-Pilot", variant="primary", size="lg")
            
        # Right Panel
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Output Response")
            output_audio = gr.Audio(label="🔊 Spatial Audio Response", autoplay=True)
            latency_meter = gr.Number(label="⚡ Execution Latency (ms)")
            
            with gr.Accordion("⚙️ Gemma 4 Execution Engine (Judge's View)", open=True):
                think_log = gr.Textbox(label="🧠 Internal Reasoning (<|think|>)", lines=4, interactive=False)
                tool_log = gr.Textbox(label="🛠️ Native Tool Execution Log (JSON)", lines=4, interactive=False)
                vram_tracker = gr.Textbox(label="🔋 Live VRAM Footprint", lines=1, interactive=False)
                
    submit_btn.click(
        fn=process_interaction,
        inputs=[input_image, input_audio, mode_dropdown],
        outputs=[output_audio, latency_meter, think_log, tool_log, vram_tracker]
    )

if __name__ == "__main__":
    # Queue is strictly needed here to prevent script-kiddies from OOM-ing the free endpoint
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
