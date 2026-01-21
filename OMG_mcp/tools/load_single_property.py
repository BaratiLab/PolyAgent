# tools/load_single_property.py
import os
import torch
from typing import Dict, Any, Optional

from tools.model_state import GLOBAL_MODEL_STATE


from OpenMacromolecularGenome.generation.vae.decoder.torch import Decoder
from OpenMacromolecularGenome.generation.vae.encoder.torch import CNNEncoder
from OpenMacromolecularGenome.generation.vae.property_predictor.torch import PropertyNetworkPredictionModule

def load_single_property_model_tool(
    session_id: str,
    property_name: str,
    parent_dir: str,
    vae_dir: str,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    MCP Tool: Load OpenMacromolecularGenome single property model.
    Loads shared VAE components from multiproperty model and single property predictor.
    """

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32

    # ---- Load VAE parameters from multiproperty model ----
    vae_parameters = torch.load(
        os.path.join("/home/vani/mcp_servers/polyAgent/OMG_mcp/src/propertyhead/finetuned_multiproperty_model", "vae_parameters_multiproperty.pth"),
        map_location=device
    )

    latent_dim = vae_parameters["latent_dimension"]

    # ---- Load encoding metadata ----
    encoding_alphabet = torch.load(os.path.join(vae_dir, "encoding_alphabet.pth"), map_location=device)
    largest_molecule_len = torch.load(os.path.join(vae_dir, "largest_molecule_len.pth"), map_location=device)

    # ---- Build shared VAE models (encoder/decoder) ----
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

    # ---- Load shared encoder/decoder weights from multiproperty model ----
    encoder.load_state_dict(torch.load(
        os.path.join("/home/vani/mcp_servers/polyAgent/OMG_mcp/src/propertyhead/finetuned_multiproperty_model", "encoder.pth"),
        map_location=device
    ))
    decoder.load_state_dict(torch.load(
        os.path.join("/home/vani/mcp_servers/polyAgent/OMG_mcp/src/propertyhead/finetuned_multiproperty_model", "decoder.pth"),
        map_location=device
    ))

    # ---- Build single property network (property_dim=1) ----
    # Single property models use a different architecture: [64, 16] hidden layers
    property_network = PropertyNetworkPredictionModule(
        latent_dim=latent_dim,
        property_dim=1,  # Single property output
        property_network_hidden_dim_list=[[64, 16]],  # Architecture for single property models
        dtype=dtype,
        weights=torch.tensor([1.0], device=device, dtype=dtype),  # Single weight
        device=device
    ).to(device)

    # ---- Load single property predictor weights ----
    property_predictor_path = os.path.join(
        "/home/vani/mcp_servers/polyAgent/OMG_mcp/src/propertyhead",
        f"finetuned_{property_name}",
        f"{property_name}_predictor.pth"
    )

    if not os.path.exists(property_predictor_path):
        raise FileNotFoundError(f"Single property predictor not found: {property_predictor_path}")

    property_network.load_state_dict(torch.load(property_predictor_path, map_location=device))

    # ---- Load single property scaler ----
    property_scaler_path = os.path.join(
        "/home/vani/mcp_servers/polyAgent/OMG_mcp/src/propertyhead",
        f"finetuned_{property_name}",
        f"{property_name}_scaler.pth"
    )

    if not os.path.exists(property_scaler_path):
        raise FileNotFoundError(f"Single property scaler not found: {property_scaler_path}")

    property_scaler = torch.load(
        property_scaler_path,
        map_location=device,
        weights_only=False
    )

    encoder.eval()
    decoder.eval()
    property_network.eval()

    # ---- Register session with single property flag ----
    GLOBAL_MODEL_STATE[session_id] = {
        "encoder": encoder,
        "decoder": decoder,
        "property_network": property_network,
        "property_scaler": property_scaler,
        "property_names": [property_name],  # Single property name
        "latent_dim": latent_dim,
        "property_dim": 1,  # Single property dimension
        "encoding_alphabet": encoding_alphabet,
        "largest_molecule_len": largest_molecule_len,
        "optimization_weights": torch.tensor([1.0], device=device, dtype=dtype),  # Single weight
        "device": device,
        "dtype": dtype,
        "is_single_property": True  # Flag to distinguish from multiproperty
    }

    return {
        "session_id": session_id,
        "latent_dim": latent_dim,
        "property_names": [property_name]
    }
