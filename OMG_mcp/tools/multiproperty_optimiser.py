# tools/optimise_latents_tool.py
from typing import Dict, Any
import numpy as np
import torch
import sys
sys.path.append("/home/vani/mcp_servers/polyAgent/OMG_mcp/tools")
from optimiser import optimize_latents_constrained

# GLOBAL_MODEL_STATE imported from model_state
from tools.model_state import GLOBAL_MODEL_STATE


def optimize_latents_constrained_tool(
    model_id: str,
    constraint_spec: Dict[str, Any],
    num_seeds: int,
    num_steps: int,
    step_size: float
) -> Dict[str, Any]:
    """
    MCP Tool: Optimize latent vectors under constraints and return real property values.
    """

    # --------- Get session-specific model state ---------
    session_data = GLOBAL_MODEL_STATE.get(model_id)
    if session_data is None:
        raise RuntimeError(f"Model session '{model_id}' not loaded.")

    # --------- Run latent-space optimization (scaled property space) ---------
    optimized_latents, optimized_properties_scaled = optimize_latents_constrained(
        session_data=session_data,
        constraint_spec=constraint_spec,
        num_seeds=num_seeds,
        num_steps=num_steps,
        step_size=step_size
    )

    # --------- Inverse-transform properties back to real space ---------
    property_scaler = session_data["property_scaler"]
    # shape: [num_candidates, property_dim]
    optimized_properties_real = property_scaler.inverse_transform(optimized_properties_scaled)

    return {
        "num_candidates": len(optimized_latents),
        "latent_shape": optimized_latents.shape,
        "properties_shape": optimized_properties_real.shape,
        "properties": optimized_properties_real,  # in real, human-readable values
        "latent_vectors": optimized_latents
    }
