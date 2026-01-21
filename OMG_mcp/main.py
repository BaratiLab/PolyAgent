# server.py

from mcp.server.fastmcp import FastMCP, Context
import torch
import sys
import uuid
sys.path.append("/home/vani/mcp_servers/polyAgent/OMG_mcp")
from importlib.resources import files
from pathlib import Path

OMG_ROOT = Path(files("OpenMacromolecularGenome"))

# from optimiser import load_multiproperty_vae
from tools.multiproperty_optimiser  import optimize_latents_constrained_tool
from tools.singleproperty_optimiser import optimize_single_property_latents_constrained_tool
from tools.load_single_property import load_single_property_model_tool
from tools.decode_latents import decode_latents_tool
from tools.validate_constraints import validate_constraints_tool
import threading
from schemas.property_registry import PROPERTY_REGISTRY

import os
from rdkit import Chem
from rdkit.Chem import RDConfig
import numpy as np
mcp = FastMCP("OMG_mcp")
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
#  adapted from  https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score 
import sascorer
import argparse





# -----------------------------
# session isolation 
# -----------------------------
MODEL_SESSIONS = {}
LOCK = threading.Lock()

@mcp.tool()
def load_model(
    ctx: Context,
    finetuned_model_path: str,
    parent_dir: str,
    vae_dir: str,
    property_names: list[str] | None = None,
    optimization_weights: list[float] | None = None,
):
    session_id = getattr(ctx, 'session_id', str(uuid.uuid4()))

    from tools.load_model import load_model_tool

    with LOCK:
        results = load_model_tool(
            session_id=session_id,
            finetuned_model_path=finetuned_model_path,
            parent_dir=parent_dir,
            vae_dir=vae_dir,
            property_names=property_names,
            device="cuda" if torch.cuda.is_available() else "cpu",
            optimization_weights=optimization_weights
        )

        MODEL_SESSIONS[session_id] = results

    return {
        "status": "loaded",
        "latent_dim": results["latent_dim"],
        "properties": results["property_names"],
    }

def format_results_table(polymers, properties_array, property_names):
    """
    polymers: list[str] or SMILES
    properties_array: np.ndarray [N, P]
    property_names: list[str]
    """

    table = []

    for i, polymer in enumerate(polymers):
        row = {
            "polymer": polymer,
            "properties": []
        }

        for j, prop in enumerate(property_names):
            meta = PROPERTY_REGISTRY.get(prop, {})
            row["properties"].append({
                "code": prop,
                "name": meta.get("name", prop),
                "value": float(properties_array[i, j]),
                "unit": meta.get("unit", "")
            })

        table.append(row)

    return table
@mcp.tool()
def sa_score_smiles(
    ctx: Context,
    smiles_list: list[str]
):
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scores.append(None)
        else:
            score = sascorer.calculateScore(mol)
            scores.append(score)
    return scores

@mcp.tool()
def decode_latents(
    ctx: Context,
    optimized_latents,
    optimized_properties,
    target_property: str,
    target_value: float,
):
    session_id = getattr(ctx, 'session_id', str(uuid.uuid4()))

    if session_id not in MODEL_SESSIONS:
        raise RuntimeError("Model not loaded for this session")

    results_dict = MODEL_SESSIONS[session_id]

    from tools.decode_latents import decode_latents_tool

    polymers = decode_latents_tool(
        optimized_latents,
        results_dict
    )

    property_names = results_dict["property_names"]

    table = format_results_table(
        polymers=polymers,
        properties_array=optimized_properties,
        property_names=property_names
    )

    return {
        "target_property": target_property,
        "target_value": target_value,
        "num_candidates": len(table),
        "results_table": table
    }

@mcp.tool()
def optimize_latents(
    ctx: Context,
    constraint_spec,
    num_seeds: int = 42,
    num_steps: int = 100,
    step_size: float = 0.1,
):
    session_id = getattr(ctx, 'session_id', str(uuid.uuid4()))

    with LOCK:
        if session_id not in MODEL_SESSIONS:
            raise RuntimeError("Model not loaded for this session")

        results_dict = MODEL_SESSIONS[session_id]

    from tools.multiproperty_optimiser import optimize_latents_constrained_tool

    return optimize_latents_constrained_tool(
        model_id=session_id,
        constraint_spec=constraint_spec,
        num_seeds=num_seeds,
        num_steps=num_steps,
        step_size=step_size
    )

def resolve_property_key(name: str) -> str:
    name = name.lower().strip()
    for key, meta in PROPERTY_REGISTRY.items():
        if name == key.lower():
            return key
        if name in meta["aliases"]:
            return key
    raise ValueError(f"Unknown property name: {name}")


@mcp.tool()
def load_single_property_model(
    ctx: Context,
    property_name: str,
    parent_dir: str,
    vae_dir: str,
):
    session_id = getattr(ctx, 'session_id', str(uuid.uuid4()))

    with LOCK:
        results = load_single_property_model_tool(
            session_id=session_id,
            property_name=property_name,
            parent_dir=parent_dir,
            vae_dir=vae_dir,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        MODEL_SESSIONS[session_id] = results

    return {
        "status": "loaded",
        "latent_dim": results["latent_dim"],
        "properties": results["property_names"],
    }

@mcp.tool()
def optimize_single_property_latents(
    ctx: Context,
    constraint_spec,
    num_seeds: int = 42,
    num_steps: int = 100,
    step_size: float = 0.1,
):
    session_id = getattr(ctx, 'session_id', str(uuid.uuid4()))

    with LOCK:
        if session_id not in MODEL_SESSIONS:
            raise RuntimeError("Model not loaded for this session")

        results_dict = MODEL_SESSIONS[session_id]

    return optimize_single_property_latents_constrained_tool(
        model_id=session_id,
        constraint_spec=constraint_spec,
        num_seeds=num_seeds,
        num_steps=num_steps,
        step_size=step_size
    )

@mcp.tool()
def optimize_single_polymer_property(
    ctx: Context,
    target_property: str,
    target_value: float,
    tolerance: float = 0.1,
    constraints: dict | None = None,
    num_seeds: int = 20,
    num_steps: int = 10,
    step_size: float = 0.1,
    lambda_constraint: float = 10.0
):
    """
    Complete pipeline for single property models: Load single property model, optimize latent vectors for target property value, and decode to polymers.

    Args:
        target_property: Name of the property to optimize (e.g., "Eea")
        target_value: Desired value for the target property (e.g., 3.2)
        tolerance: Acceptable deviation from target value
        constraints: Optional additional property constraints as dict of {property_name: [min, max]}
        num_seeds: Number of latent seeds for optimization
        num_steps: Number of optimization steps per seed
        step_size: Gradient step size for optimization
        lambda_constraint: Penalty weight for constraint violations
    """
    session_id = getattr(ctx, 'session_id', str(uuid.uuid4()))

    # Hardcoded paths - update these if needed
    parent_dir = "/home/vani/mcp_servers/polyAgent/OMG_mcp"
    vae_dir = "/home/vani/mcp_servers/polyAgent/OMG_mcp/src/all/vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay/divergence_weight_4.345_latent_dim_152_learning_rate_0.002"
    # Step 1: Load the single property model
    with LOCK:
        model_results = load_single_property_model_tool(
            session_id=session_id,
            property_name=target_property,
            parent_dir=parent_dir,
            vae_dir=vae_dir,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        MODEL_SESSIONS[session_id] = model_results

    # Step 2: Create constraint specification
    constraint_spec = {
        "target_property": target_property,
        "target_value": target_value,
        "tolerance": tolerance,
        "constraints": constraints or {},
        "lambda_constraint": lambda_constraint
    }

    # Step 3: Run optimization
    optimization_results = optimize_single_property_latents_constrained_tool(
        model_id=session_id,
        constraint_spec=constraint_spec,
        num_seeds=num_seeds,
        num_steps=num_steps,
        step_size=step_size
    )

    # Step 4: Decode the optimized latents and select top 5 closest to target
    decoded_results = decode_latents_tool(
        session_id=session_id,
        optimized_latents=optimization_results["latent_vectors"],
        optimized_properties=optimization_results["properties"],
        target_property=target_property,
        target_value=target_value
    )

    # Return combined results
    return {
        "model_loaded": {
            "latent_dim": model_results["latent_dim"],
            "properties": model_results["property_names"]
        },
        "optimization": {
            "num_candidates": optimization_results["num_candidates"],
            "latent_shape": optimization_results["latent_shape"],
            "properties_shape": optimization_results["properties_shape"]
        },
        "decoded_polymers": decoded_results
    }

@mcp.tool()
def optimize_polymer_properties(
    ctx: Context,
    target_property: str,
    target_value: float,
    tolerance: float = 0.1,
    constraints: dict | None = None,
    num_seeds: int = 20,
    num_steps: int = 10,
    step_size: float = 0.1,
    lambda_constraint: float = 10.0
):
    """
    Complete pipeline: Load model, optimize latent vectors for target property value, and decode to polymers.

    Args:
        target_property: Name of the property to optimize (e.g., "Eea")
        target_value: Desired value for the target property (e.g., 3.2)
        save_directory: Directory to save decoded polymer results
        tolerance: Acceptable deviation from target value
        constraints: Optional additional property constraints as dict of {property_name: [min, max]}
        num_seeds: Number of latent seeds for optimization
        num_steps: Number of optimization steps per seed
        step_size: Gradient step size for optimization
        lambda_constraint: Penalty weight for constraint violations
    """
    import uuid
    # session_id = str(uuid.uuid4())  # Generate unique session ID
    session_id = getattr(ctx, 'session_id', str(uuid.uuid4()))

    # Hardcoded paths - update these if needed
    finetuned_model_path = "/home/vani/mcp_servers/OMG_mcp/src/propertyhead/finetuned_multiproperty_model"
    parent_dir = "/home/vani/mcp_servers/OMG_mcp"
    vae_dir = "/home/vani/mcp_servers/OMG_mcp/src/all/vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay/divergence_weight_4.345_latent_dim_152_learning_rate_0.002"

    # Step 1: Load the model
    from tools.load_model import load_model_tool

    with LOCK:
        model_results = load_model_tool(
            session_id=session_id,
            finetuned_model_path=finetuned_model_path,
            parent_dir=parent_dir,
            vae_dir=vae_dir,
            property_names=None,  # Use defaults from model
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        MODEL_SESSIONS[session_id] = model_results

    # Step 2: Create constraint specification
    constraint_spec = {
        "target_property": target_property,
        "target_value": target_value,
        "tolerance": tolerance,
        "constraints": constraints or {},
        "lambda_constraint": lambda_constraint
    }

    # Step 3: Run optimization
    from tools.multiproperty_optimiser import optimize_latents_constrained_tool

    optimization_results = optimize_latents_constrained_tool(
        model_id=session_id,
        constraint_spec=constraint_spec,
        num_seeds=num_seeds,
        num_steps=num_steps,
        step_size=step_size
    )

    # Step 4: Decode the optimized latents and select top 5 closest to target
    from tools.decode_latents import decode_latents_tool

    decoded_results = decode_latents_tool(
        session_id=session_id,
        optimized_latents=optimization_results["latent_vectors"],
        optimized_properties=optimization_results["properties"],
        
        target_property=target_property,
        target_value=target_value
    )

    # Return combined results
    return {
        "model_loaded": {
            "latent_dim": model_results["latent_dim"],
            "properties": model_results["property_names"]
        },
        "optimization": {
            "num_candidates": optimization_results["num_candidates"],
            "latent_shape": optimization_results["latent_shape"],
            "properties_shape": optimization_results["properties_shape"]
        },
        "decoded_polymers": decoded_results
    }


if __name__ == "__main__":
    
    mcp.run(transport="stdio")
