# tools/load_model.py
import os
import torch
from typing import Dict, Any, List, Optional

from tools.model_state import GLOBAL_MODEL_STATE

import sys
#sys.path.append('/home/vani/omgfine/OpenMacromolecularGenome/generation')
from OpenMacromolecularGenome.generation.vae.decoder.torch import Decoder
from OpenMacromolecularGenome.generation.vae.encoder.torch import CNNEncoder
from OpenMacromolecularGenome.generation.vae.property_predictor.torch import PropertyNetworkPredictionModule


def load_model_tool(
    session_id: str,
    finetuned_model_path: str,
    parent_dir: str,
    vae_dir: str,
    property_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    optimization_weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    MCP Tool: Load OpenMacromolecularGenome multiproperty model.
    """

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32

    # ---- Load VAE parameters ----
    vae_parameters = torch.load(
        os.path.join(finetuned_model_path, "vae_parameters_multiproperty.pth"),
        map_location=device
    )

    latent_dim = vae_parameters["latent_dimension"]
    property_dim = vae_parameters["property_dim"]

    # Resolve property names
    saved_names = vae_parameters.get("property_names")
    if property_names is not None:
        if len(property_names) != property_dim:
            raise ValueError("property_names length mismatch")
        vae_parameters["property_names"] = property_names
        torch.save(
            vae_parameters,
            os.path.join(finetuned_model_path, "vae_parameters_multiproperty.pth")
        )
        resolved_names = property_names
    else:
        resolved_names = saved_names

    # ---- Load encoding metadata ----
    encoding_alphabet = torch.load(os.path.join(vae_dir, "encoding_alphabet.pth"), map_location=device)
    largest_molecule_len = torch.load(os.path.join(vae_dir, "largest_molecule_len.pth"), map_location=device)

    # ---- Build models ----
    encoder = CNNEncoder(
        in_channels=vae_parameters["encoder_in_channels"],
        feature_dim=vae_parameters["encoder_feature_dim"],
        convolution_channel_dim=vae_parameters["encoder_convolution_channel_dim"],
        kernel_size=vae_parameters["encoder_kernel_size"],
        layer_1d=vae_parameters["encoder_layer_1d"],
        layer_2d=vae_parameters["encoder_layer_2d"],
        latent_dimension=latent_dim
    ).to(device)

    decoder = Decoder(
        input_size=vae_parameters["decoder_input_dimension"],
        num_layers=vae_parameters["decoder_num_gru_layers"],
        hidden_size=latent_dim,
        out_dimension=vae_parameters["decoder_output_dimension"],
        bidirectional=vae_parameters["decoder_bidirectional"]
    ).to(device)

    property_network = PropertyNetworkPredictionModule(
        latent_dim=latent_dim,
        property_dim=property_dim,
        property_network_hidden_dim_list=vae_parameters["property_network_hidden_dim_list"],
        dtype=dtype,
        device=device,
        weights=vae_parameters["property_weights"]
    ).to(device)

    # ---- Load weights ----
    encoder.load_state_dict(torch.load(os.path.join(finetuned_model_path, "encoder.pth"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(finetuned_model_path, "decoder.pth"), map_location=device))
    property_network.load_state_dict(
        torch.load(os.path.join(finetuned_model_path, "property_predictor_multiproperty.pth"), map_location=device)
    )

    encoder.eval()
    decoder.eval()
    property_network.eval()

    # ---- Load scaler ----
    property_scaler = torch.load(
        os.path.join(finetuned_model_path, "property_scaler.pth"),
        map_location=device,
        weights_only=False
    )

    # Set default optimization weights if not provided
    if optimization_weights is None:
        optimization_weights = [1.0] * property_dim  # Default equal weights
    elif len(optimization_weights) != property_dim:
        raise ValueError(f"optimization_weights length {len(optimization_weights)} does not match property_dim {property_dim}")
    optimization_weights_tensor = torch.tensor(optimization_weights, device=device, dtype=dtype)

    # ---- Register session ----
    GLOBAL_MODEL_STATE[session_id] = {
        "encoder": encoder,
        "decoder": decoder,
        "property_network": property_network,
        "property_scaler": property_scaler,
        "property_names": resolved_names,
        "latent_dim": latent_dim,
        "property_dim": property_dim,
        "encoding_alphabet": encoding_alphabet,
        "largest_molecule_len": largest_molecule_len,
        "optimization_weights": optimization_weights_tensor,
        "device": device,
        "dtype": dtype
    }

    return {
        "session_id": session_id,
        "latent_dim": latent_dim,
        "property_names": resolved_names
    }


