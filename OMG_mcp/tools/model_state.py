# tools/model_state.py
from typing import Dict, Any

# session_id → model objects
GLOBAL_MODEL_STATE: Dict[str, Dict[str, Any]] = {}
