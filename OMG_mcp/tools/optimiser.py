# optimiser.py

import torch
import numpy as np
from typing import Dict, Any, Tuple

from tools.model_state import GLOBAL_MODEL_STATE

# for multiproperty optimization
def optimize_latents_constrained(
    session_data: Dict[str, Any],
    constraint_spec: Dict[str, Any],
    num_seeds: int,
    num_steps: int,
    step_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constrained latent-space optimization using penalty-based gradient ascent.

    Optimization is performed in SCALED property space.
    Returned properties are STILL SCALED.
    """

    # --------- Load global model state (injected by MCP server) ---------

    global GLOBAL_MODEL_STATE
    


    property_network = session_data["property_network"]
    property_scaler = session_data["property_scaler"]
    property_names = session_data["property_names"]
    latent_dim = session_data["latent_dim"]
    device = session_data["device"]
    dtype = session_data["dtype"]

    # --------- Parse constraint spec ---------
    target_property = constraint_spec["target_property"]
    target_value = constraint_spec["target_value"]
    tolerance = constraint_spec["tolerance"]
    constraints = constraint_spec.get("constraints", {})
    lambda_constraint = constraint_spec.get("lambda_constraint", 10.0)

    name_to_idx = {name: i for i, name in enumerate(property_names)}

    target_idx = name_to_idx[target_property]

    # --------- SCALE target & constraint values ---------
    # property_scaler expects shape [N, property_dim]
    dummy = np.zeros((1, len(property_names)))

    # Scale target value
    dummy[0, target_idx] = target_value
    scaled_target = property_scaler.transform(dummy)[0, target_idx]

    # Scale constraint ranges
    scaled_constraints = {}
    for prop, (low, high) in constraints.items():
        idx = name_to_idx[prop]

        dummy_low = np.zeros((1, len(property_names)))
        dummy_high = np.zeros((1, len(property_names)))

        dummy_low[0, idx] = low
        dummy_high[0, idx] = high

        low_s = property_scaler.transform(dummy_low)[0, idx]
        high_s = property_scaler.transform(dummy_high)[0, idx]

        scaled_constraints[idx] = (low_s, high_s)

    # --------- Initialize latent seeds ---------
    torch.manual_seed(42)
    z_seeds = torch.randn(num_seeds, latent_dim, device=device, dtype=dtype)

    optimized_latents = []
    optimized_properties = []

    # --------- Optimization loop ---------
    for seed_idx in range(num_seeds):
        z = z_seeds[seed_idx].clone().detach().requires_grad_(True)

        best_obj = -float("inf")
        best_z = None

        for _ in range(num_steps):
            network_output = property_network(z.unsqueeze(0))
            # Handle case where property_network returns a tuple
            if isinstance(network_output, tuple):
                props = network_output[0].squeeze(0)  # SCALED
            else:
                props = network_output.squeeze(0)  # SCALED

            # Target attraction (stay within ± tolerance)
            target_loss = - (props[target_idx] - scaled_target) ** 2

            # Constraint penalties
            penalty = 0.0
            for idx, (low_s, high_s) in scaled_constraints.items():
                penalty += torch.relu(low_s - props[idx]) ** 2
                penalty += torch.relu(props[idx] - high_s) ** 2

            # Final objective
            objective = target_loss - lambda_constraint * penalty

            if objective.item() > best_obj:
                best_obj = objective.item()
                best_z = z.detach().clone()

            # Backprop
            if z.grad is not None:
                z.grad.zero_()

            objective.backward()

            with torch.no_grad():
                grad = z.grad
                z += step_size * grad / (torch.norm(grad) + 1e-8)

        # --------- Store best result ---------
        final_z = best_z
        final_network_output = property_network(final_z.unsqueeze(0))
        # Handle case where property_network returns a tuple
        if isinstance(final_network_output, tuple):
            final_props = final_network_output[0]
        else:
            final_props = final_network_output

        optimized_latents.append(final_z.cpu().numpy())
        optimized_properties.append(final_props.detach().cpu().numpy())

    return (
        np.array(optimized_latents),
        np.array(optimized_properties).squeeze()
    )
