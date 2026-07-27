import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# Gemma 4 Native Tool Schemas (Pydantic)
# ==========================================

class ParsePrescriptionLabel(BaseModel):
    """
    Schema for parsing prescription label data.
    """
    medication_name: str = Field(description="The extracted name of the medication.")
    dosage: str = Field(description="The extracted dosage instructions (e.g., '500mg twice a day').")
    expiry_date: str = Field(description="The extracted expiration date (e.g., 'YYYY-MM-DD' or 'MM/YYYY').")

class VerifyMedicationExpiry(BaseModel):
    """
    Schema for verifying if a medication is expired.
    """
    expiry_date: str = Field(description="The expiration date to verify (e.g., 'YYYY-MM-DD' or 'MM/YYYY').")

class TriggerHapticFeedback(BaseModel):
    """
    Schema for triggering physical device haptic feedback.
    """
    pattern: str = Field(description="The haptic pattern to trigger. Must be 'warning', 'stop', or 'confirm'.")


# ==========================================
# Tool Dispatcher
# ==========================================

class ToolDispatcher:
    def __init__(self):
        self.tools = {
            "parse_prescription_label": self._parse_prescription_label,
            "verify_medication_expiry": self._verify_medication_expiry,
            "trigger_haptic_feedback": self._trigger_haptic_feedback
        }

    def dispatch(self, tool_call_json: str) -> str:
        """
        Parses the JSON tool call from Gemma 4, executes the corresponding local python logic,
        and returns the formatted JSON result.
        """
        try:
            call_data = json.loads(tool_call_json)
            tool_name = call_data.get("name")
            arguments = call_data.get("arguments", {})
            
            if tool_name not in self.tools:
                error_msg = f"Unknown tool: {tool_name}"
                logger.error(error_msg)
                return self._format_response(tool_name, "error", error_msg)
                
            logger.info(f"Executing tool '{tool_name}' with args: {arguments}")
            
            # Execute the local python function
            result = self.tools[tool_name](**arguments)
            return self._format_response(tool_name, "success", result, arguments)
            
        except json.JSONDecodeError:
            error_msg = "Invalid JSON tool call provided by the model."
            logger.error(error_msg)
            return self._format_response("unknown", "error", error_msg)
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(error_msg)
            return self._format_response(call_data.get("name", "unknown"), "error", {"error": error_msg})

    def _format_response(self, tool_name: str, status: str, data: Any, arguments: dict = None) -> str:
        """Formats the result into a styled HTML card with status icons and JSON fallback."""
        if status == "error":
            icon, title, color = "🔴", "ERROR", "#EF4444"
            desc = data.get("error", "Unknown error")
            
        elif tool_name == "parse_prescription_label":
            icon, title, color = "🟢", "SUCCESS", "#10B981"
            med = data.get("data", {}).get("medication_name", "Unknown")
            dos = data.get("data", {}).get("dosage", "")
            desc = f"Extracted: {med} {dos}."
            
        elif tool_name == "trigger_haptic_feedback":
            icon, title, color = "🟡", "HAPTIC PULSE", "#F59E0B"
            pat = arguments.get("pattern", "Unknown") if arguments else "Unknown"
            desc = f"Signal Sent: {pat.capitalize()} (Vibration)."
            
        elif tool_name == "verify_medication_expiry":
            if data.get("is_safe"):
                icon, title, color = "🟢", "SAFETY CHECK", "#10B981"
            else:
                icon, title, color = "🔴", "SAFETY WARNING", "#EF4444"
            desc = data.get("warning", "")
            
        else:
            icon, title, color = "⚪", "INFO", "#94A3B8"
            desc = str(data)

        # Build an HTML card
        html = f'''
        <div style="background-color: #080C14; border-left: 4px solid {color}; padding: 12px; margin-top: 8px; border-radius: 4px; font-family: monospace;">
            <div style="color: {color}; font-weight: bold; margin-bottom: 4px;">
                {icon} {title}: {tool_name}()
            </div>
            <div style="color: #E2E8F0; padding-left: 24px;">
                &rarr; {desc}
            </div>
            <div style="color: #475569; font-size: 0.8em; padding-left: 24px; margin-top: 4px;">
                [RAW JSON] {json.dumps(data)}
            </div>
        </div>
        '''
        return html

    # --- Tool Implementations ---
    
    def _parse_prescription_label(self, medication_name: str, dosage: str, expiry_date: str) -> Dict[str, Any]:
        """Stores or processes the parsed label data locally."""
        return {
            "message": "Prescription logged successfully.",
            "data": {
                "medication_name": medication_name,
                "dosage": dosage,
                "expiry_date": expiry_date
            }
        }

    def _verify_medication_expiry(self, expiry_date: str) -> Dict[str, Any]:
        """Calculates expiration safety flag."""
        try:
            # Simplified parsing logic for the demonstration
            import re
            year_match = re.search(r'20\d{2}', expiry_date)
            if year_match:
                year = int(year_match.group(0))
                current_year = datetime.now().year
                if year < current_year:
                    return {"is_safe": False, "warning": f"Medication expired in {year}. DO NOT USE."}
                elif year == current_year:
                    return {"is_safe": True, "warning": "Medication expires this year. Check the month."}
                else:
                    return {"is_safe": True, "warning": "Medication is safe to use."}
            else:
                return {"is_safe": False, "warning": "Could not parse expiration year. Proceed with caution."}
        except Exception as e:
            return {"is_safe": False, "warning": f"Parse error: {str(e)}"}

    def _trigger_haptic_feedback(self, pattern: str) -> Dict[str, Any]:
        """Simulates GPIO haptic motor triggers for hardware deployment."""
        valid_patterns = ["warning", "stop", "confirm"]
        if pattern not in valid_patterns:
            return {"success": False, "message": f"Invalid pattern. Must be one of: {valid_patterns}"}
            
        # Hardware GPIO logic would go here (e.g. Raspberry Pi Zero W / RPi.GPIO)
        logger.info(f"[HARDWARE] Haptic motor triggered: {pattern.upper()}")
        return {"success": True, "message": f"Haptic pattern '{pattern}' executed."}

if __name__ == "__main__":
    # Test the dispatcher
    print("--- Testing Tool Dispatcher ---")
    dispatcher = ToolDispatcher()
    
    test_json = json.dumps({
        "name": "verify_medication_expiry",
        "arguments": {
            "expiry_date": "2020-05"
        }
    })
    
    print(f"Incoming Model Call:\n{test_json}")
    result = dispatcher.dispatch(test_json)
    print(f"\nExecution Result:\n{result}")
