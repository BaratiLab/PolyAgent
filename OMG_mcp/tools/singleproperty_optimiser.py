# tools/singleproperty_optimiser.py
from typing import Dict, Any
import numpy as np
import torch
import sys
sys.path.append("/home/vani/mcp_servers/polyAgent/OMG_mcp/tools")
from optimiser import optimize_latents_constrained

# GLOBAL_MODEL_STATE imported from model_state
from tools.model_state import GLOBAL_MODEL_STATE

def optimize_single_property_latents_constrained_tool(
    model_id: str,
    constraint_spec: Dict[str, Any],
    num_seeds: int,
    num_steps: int,
    step_size: float
) -> Dict[str, Any]:
    """
    MCP Tool: Optimize latent vectors under constraints for single property models.
    Returns real property values.
    """

    # --------- Get session-specific model state ---------
    session_data = GLOBAL_MODEL_STATE.get(model_id)
    if session_data is None:
        raise RuntimeError(f"Model session '{model_id}' not loaded.")

    # Check if this is a single property model
    if not session_data.get("is_single_property", False):
        raise RuntimeError(f"Model session '{model_id}' is not a single property model.")
    property_name = session_data["property_names"][0]
    session_data["use_latent_mean"] = True

    # --------- Run latent-space optimization (scaled property space) ---------
    optimized_latents, optimized_properties_scaled = optimize_latents_constrained(
        session_data=session_data,
        constraint_spec=constraint_spec,
        num_seeds=num_seeds,
        num_steps=num_steps,
        step_size=step_size
    )
    if "constraints" in constraint_spec and constraint_spec["constraints"]:
        raise ValueError(
        f"Single-property optimizer received extra constraints: "
        f"{constraint_spec['constraints']}. "
        f"Only target_property is allowed."
    )


    # --------- Inverse-transform properties back to real space ---------
    property_scaler = session_data["property_scaler"]
    # shape: [num_candidates, property_dim] - property_dim=1 for single property
    optimized_properties_real = np.asarray(optimized_properties_scaled)

    if optimized_properties_scaled.ndim == 1:
        optimized_properties_scaled = optimized_properties_scaled.reshape(-1, 1)

    optimized_properties_real = property_scaler.inverse_transform(
        optimized_properties_scaled
    )

    return {
        "num_candidates": len(optimized_latents),
        "latent_shape": optimized_latents.shape,
        "properties_shape": optimized_properties_real.shape,
        "properties": optimized_properties_real,  # in real, human-readable values
        "latent_vectors": optimized_latents
    }
