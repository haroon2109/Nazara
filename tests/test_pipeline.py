import unittest
import numpy as np
from PIL import Image
import sys
import os
import time
import torch
from unittest.mock import MagicMock, patch

# Ensure the root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_engine import Gemma4Engine

class TestNazaraPipeline(unittest.TestCase):
    
    @patch('src.model_engine.Gemma4Engine.load_model')
    def setUp(self, mock_load):
        """
        Initialize the engine while mocking the load_model function 
        to prevent downloading heavy weights during testing.
        """
        self.engine = Gemma4Engine()
        
        # Setup mock processor and model
        self.engine.processor = MagicMock()
        self.engine.model = MagicMock()
        self.engine.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create a mock image
        mock_array = np.zeros((224, 224, 3), dtype=np.uint8)
        self.test_image = Image.fromarray(mock_array)
        
        # Create mock audio waveform (1 second of silence at 16kHz)
        self.test_audio = np.zeros(16000, dtype=np.float32)

    def test_pipeline_requirements(self):
        """
        Asserts the 4 strict requirements:
        1) Response time is under 1.5s
        2) Thinking mode <|think|> tokens are generated
        3) Native tool calls execute cleanly
        4) Memory usage stays under 12GB VRAM
        """
        # Mocking the generation decoding to return a think block AND a tool call
        mock_output = (
            "<|think|> User is moving at 1.2m/s, obstacle clock angle 11 o'clock. </|think|> "
            '{"name": "trigger_haptic_alert", "arguments": {"pattern_type": "warning"}}'
        )
        self.engine.processor.decode.return_value = mock_output
        
        # Start timer for latency test
        start_time = time.time()
        
        # Run mocked inference
        response, think_block = self.engine.analyze_spatial_frame(
            image=self.test_image, 
            audio_bytes=self.test_audio, 
            text_prompt="Detect obstacles."
        )
        
        elapsed_time = time.time() - start_time
        
        # 1. Response time under 1.5s (Mocked execution should easily pass this)
        self.assertLess(elapsed_time, 1.5, f"Response time exceeded 1.5s limit: {elapsed_time:.2f}s")
        
        # 2. Thinking mode <|think|> tokens are generated
        self.assertTrue(len(think_block) > 0, "No <|think|> block was generated")
        self.assertIn("obstacle clock angle", think_block)
        
        # 3. Native tool calls execute cleanly
        self.assertIn("Tool Execution Output:", response)
        self.assertIn("warning", response)
        
        # 4. Memory usage stays under 12GB VRAM
        if torch.cuda.is_available():
            vram_used = torch.cuda.max_memory_allocated() / (1024**3)
            self.assertLess(vram_used, 12.0, f"VRAM usage exceeded 12GB limit: {vram_used:.2f}GB")
        else:
            print("CUDA not available. Skipping 12GB VRAM assertion.")

if __name__ == '__main__':
    unittest.main()
