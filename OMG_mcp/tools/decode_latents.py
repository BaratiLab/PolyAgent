# tools/decode_latents_tool.py
from typing import Dict, Any
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as f

from rdkit import Chem

from tools.model_state import GLOBAL_MODEL_STATE  # ensures consistent session access

import OpenMacromolecularGenome.generation.selfies as sf


def decode_latents_to_polymers(optimized_latents: np.ndarray, optimized_properties: np.ndarray,
                               results_dict: Dict[str, Any],
                               target_property: str, target_value: float) -> list:
    """
    Decodes latent vectors into polymers, calculates final properties, and selects top 5 closest to target value.

    Args:
        optimized_latents: Array of optimized latent vectors.
        optimized_properties: Array of predicted (real) properties for the latents.
        results_dict: Dictionary containing decoder, scaler, alphabet, etc.
        save_directory: Directory to save the output files.
        target_property: Name of the property to optimize for closeness.
        target_value: Target value for the property.

    Returns:
        List of dicts containing the top 5 polymers closest to target value with structure and other properties.
    """
    decoder = results_dict['decoder']
    property_scaler = results_dict['property_scaler']
    encoding_alphabet = results_dict['encoding_alphabet']
    largest_molecule_len = results_dict['largest_molecule_len']
    latent_dim = results_dict['latent_dim']
    property_dim = results_dict['property_dim']
    property_names = results_dict['property_names']
    optimization_weights = results_dict['optimization_weights'].cpu().numpy()
    device = results_dict['device']
    dtype = results_dict['dtype']

    print("Decoding optimized latent vectors to polymers...")
    generated_polymers = []
    valid_polymers = []

    with torch.no_grad():
        for latent_vec in optimized_latents:
            z_tensor = torch.tensor(latent_vec, dtype=dtype, device=device)

            # VAE decoder generation
            hidden = decoder.init_hidden(z_tensor.unsqueeze(0))
            x_input = torch.zeros(size=(1, 1, len(encoding_alphabet)), dtype=dtype, device=device)
            gathered_indices = []

            for _ in range(largest_molecule_len):
                out_one_hot_line, hidden = decoder(x=x_input, hidden=hidden)
                x_hat_prob = f.softmax(out_one_hot_line, dim=-1)
                x_hat_indices = x_hat_prob.argmax(dim=-1)
                x_input = f.one_hot(x_hat_indices, num_classes=len(encoding_alphabet)).to(torch.float)
                gathered_indices.append(x_hat_indices.squeeze().item())

            # Convert indices to SMILES via SELFIES
            gathered_atoms = ''.join([encoding_alphabet[idx] for idx in gathered_indices])
            generated_selfies = '[*]' + gathered_atoms.replace('[nop]', '')
            smiles_generated = None
            is_valid = False
            try:
                smiles_generated = sf.decoder(generated_selfies)
                mol = Chem.MolFromSmiles(smiles_generated)
                if mol is not None:
                    smiles_generated = Chem.MolToSmiles(mol)
                    is_valid = True
            except:
                pass  # Keep defaults: smiles_generated=None, is_valid=False

            generated_polymers.append(smiles_generated)
            valid_polymers.append(is_valid)

    # Create results dataframe
    results_df = pd.DataFrame({
        'polymer_smiles': generated_polymers,
        'is_valid': valid_polymers,
    })

    # Properties are already in real space
    optimized_properties_unscaled = optimized_properties
    for i in range(property_dim):
        results_df[property_names[i]] = optimized_properties_unscaled[:, i]

    # Filter valid polymers
    valid_mask = results_df['is_valid']
    results_df_valid = results_df[valid_mask]

    # Get target property index
    target_idx = property_names.index(target_property)

    # Calculate distance to target value
    distances = np.abs(results_df_valid[property_names[target_idx]] - target_value)
    results_df_valid = results_df_valid.assign(distance=distances)

    # Select top 5 closest to target value
    top_5_df = results_df_valid.nsmallest(5, 'distance')

    # Prepare output: list of dicts with structure and other 4 properties
    other_properties = [p for p in property_names if p != target_property]
    top_polymers = []
    for _, row in top_5_df.iterrows():
        polymer_dict = {'structure': row['polymer_smiles']}
        for prop in other_properties:
            polymer_dict[prop] = float(row[prop])
        top_polymers.append(polymer_dict)

    print(f"Optimization complete! Generated {len(results_df_valid)} valid polymers.")
    print(f"Selected top 5 polymers closest to target value {target_value} for property {target_property}.")
    print("Results stored as JSON structure.")

    return top_polymers







        
        
def decode_latents_tool(
    session_id: str,
    optimized_latents: "np.ndarray",
    optimized_properties: "np.ndarray",
    #save_directory: str,
    target_property: str,
    target_value: float
) -> Dict[str, Any]:
    """
    MCP Tool: Decode latent vectors into polymers using the real property values
    returned from optimize_latents_constrained_tool, selecting top 5 closest to target.
    """

    # Ensure session-specific model objects exist
    results_dict = GLOBAL_MODEL_STATE.get(session_id)
    if results_dict is None:
        raise RuntimeError(f"Model session '{session_id}' not loaded in GLOBAL_MODEL_STATE.")

    # Decode latents into polymer structures and select top 5
    top_polymers = decode_latents_to_polymers(
        optimized_latents=optimized_latents,
        optimized_properties=optimized_properties,  # already inverse-transformed
        results_dict=results_dict,
        #save_directory=save_directory,
        target_property=target_property,
        target_value=target_value
    )

    return {
        "num_selected": len(top_polymers),
        "target_property": target_property,
        "target_value": target_value,
        #"output_path": save_directory,
        "top_polymers": top_polymers  # list of dicts with structure and other properties
    }
