# main.py — MCP server for TransPolymer PE_I inference
# FastMCP 2.x
from mcp.server.fastmcp import FastMCP, Context

import os, sys, csv, json, traceback, threading, subprocess, shlex
from pathlib import Path
from typing import List, Optional

import torch


from transformers import RobertaModel
from transpolymer_pretrained.core import downstream
from transpolymer_pretrained.core.PolymerSmilesTokenization import PolymerSmilesTokenizer
from importlib.resources import files
from pathlib import Path

PKG_ROOT: Path = Path(files("transpolymer_pretrained"))

# ---- multi-property support ----
PROPERTIES = ["Eea", "Egb", "EPS", "PE_I", "OPV"]

PROPERTY_CONFIGS = {
    "Eea": PKG_ROOT / "configs" / "config_Eea.yaml",
    "Egb": PKG_ROOT / "configs" / "config_Egb.yaml",
    "EPS": PKG_ROOT / "configs" / "config_EPS.yaml",
    "PE_I": PKG_ROOT / "configs" / "config_pe.yaml",
    "OPV": PKG_ROOT / "configs" / "config_opv.yaml",
}

DEFAULT_PROP_CKPTS = {
    prop: PKG_ROOT / "ckpt" / f"{prop}_best_model.pt"
    for prop in PROPERTIES
}


_prop_models: dict[str, torch.nn.Module] = {}
_prop_scalers: dict[str, object] = {}
_prop_tokenizers: dict[str, object] = {}
_prop_maxlens: dict[str, int] = {}



# ===================================================================



mcp = FastMCP("TransPolymer_mcp")
_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_max_len = 256
_scaler = None
_lock = threading.Lock()
_loaded_via_import = False  # if False, we'll use subprocess fallback

# --- add with your other helpers ---
def _resolve_rel(base: Path, maybe_path: str | None) -> str | None:
    if not maybe_path:
        return None
    p = Path(maybe_path)
    return str(p if p.is_absolute() else (base / p))

def _build_property(prop: str):
    """
    Build backbone + tokenizer + regression head for a single property,
    using its own config and vocab. Called lazily on first use.
    """
    if prop not in PROPERTIES:
        raise ValueError(f"Unknown property '{prop}'. Expected one of {PROPERTIES}.")
    if prop in _prop_models:
        return  # already built

    cfg_path = PROPERTY_CONFIGS[prop]
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config for {prop} not found at {cfg_path}")

    print(f"[mcp] Building model for property {prop} using config {cfg_path}")
    raw_cfg = _load_yaml(cfg_path)
    cfg = _normalize_cfg_paths(raw_cfg, PKG_ROOT)



    max_len = int(cfg.get("blocksize", 256))
    model_dir = cfg.get("model_path", str(PKG_ROOT / "ckpt"))
    drop_rate = float(cfg.get("drop_rate", 0.1))

    # --- 1) Backbone for THIS property ---
    print(f"[mcp:{prop}] Loading backbone from {model_dir}")
    downstream.PretrainedModel = RobertaModel.from_pretrained(
        model_dir, local_files_only=True
    )

    # --- 2) Tokenizer for THIS property ---
    tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=max_len)
    _maybe_add_supp_vocab(tokenizer, cfg)  # uses this prop's vocab_sup_file if present
    downstream.tokenizer = tokenizer

    # --- 3) Regression head + checkpoint ---
    model = downstream.DownstreamRegression(drop_rate=drop_rate)

    ckpt = (
        os.environ.get(f"TRANSPOLYMER_{prop}_CKPT")
        or cfg.get("best_model_path")
        or str(DEFAULT_PROP_CKPTS[prop])
    )
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"[mcp:{prop}] Checkpoint not found: {ckpt}")

    _load_checkpoint(model, ckpt, map_location="cpu")
    scaler = _load_scaler_for_ckpt(
        ckpt,
        override_path=os.environ.get(f"TRANSPOLYMER_{prop}_SCALER", None),
    )

    model = model.to(_device).double().eval()

    # --- 4) Cache per-property objects ---
    _prop_models[prop] = model
    _prop_scalers[prop] = scaler
    _prop_tokenizers[prop] = tokenizer
    _prop_maxlens[prop] = max_len

    print(f"[mcp:{prop}] Ready (max_len={max_len}, vocab_size={len(tokenizer)})")


def _normalize_cfg_paths(cfg: dict, base: Path) -> dict:
    """Return a shallow-copied cfg with all known file/dir fields absolutized."""
    out = dict(cfg)
    # folders / files that commonly appear in your configs
    keys = [
        "model_path",          # dir with pretrained backbone (ckpt)
        "vocab_sup_file",      # csv with extra tokens
        "best_model_path",     # finetuned checkpoint .pt
        "save_path",           # training save path (fallback)
        "train_file",
        "test_file",
    ]
    for k in keys:
        if k in out and out[k]:
            out[k] = _resolve_rel(base, out[k])
    return out

def _chdir_package_root():
    # Some libs assume CWD ~ project root (optional but helps)
    try:
        os.chdir(str(PKG_ROOT))
    except Exception:
        pass


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str | Path, map_location="cpu"):
    ckpt_path = str(ckpt_path)
    print(f"[mcp] Loading state_dict from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=map_location)
    # Support both pure state_dict and {"state_dict": ...} and {"model": ...}
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)


def _load_scaler_for_ckpt(ckpt_path: str | Path, override_path: str | None = None):
    import joblib

    ckpt_path = str(ckpt_path)
    scaler_path = override_path or ckpt_path.replace(".pt", ".scaler.pkl")
    if os.path.exists(scaler_path):
        print(f"[mcp] Loading scaler from {scaler_path}")
        return joblib.load(scaler_path)
    print(f"[mcp] No scaler found for {ckpt_path} (expected {scaler_path})")
    return None



def _load_yaml(path: Path):
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def _maybe_add_supp_vocab(tokenizer, cfg: dict):
    if not cfg.get("add_vocab_flag", False):
        return
    sup_path = cfg.get("vocab_sup_file", "")
    if not sup_path or not os.path.exists(sup_path):
        print(f"[mcp] add_vocab_flag=True but file not found: {sup_path}. Skipping.")
        return
    with open(sup_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        vocab_sup = [row[0] for row in reader if row]
    if vocab_sup:
        added = tokenizer.add_tokens(vocab_sup)
        print(f"[mcp] Added {added} supplementary tokens from {sup_path}")

def _safe_list_str(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(s) for s in x]

def _startup_load():
    """Lightweight startup: we will lazily build property models on first use."""
    global _loaded_via_import, _device

    _chdir_package_root()
    _device = os.environ.get("TRANSPOLYMER_DEVICE", _device)

    try:
        # sanity check imports so we fail early if something is badly broken
        import transformers  # noqa: F401
        from transpolymer_pretrained.core import downstream  # noqa: F401
        from transpolymer_pretrained.core import PolymerSmilesTokenization  # noqa: F401

        _loaded_via_import = True
        print(f"✅ MCP: Lazy per-property loading enabled (device={_device}).")
    except Exception:
        print("❌ MCP: Failed basic imports; cannot do in-process inference.")
        traceback.print_exc()
        _loaded_via_import = False


# def _predict_inprocess(smiles: List[str]) -> List[float]:
#     """Use the already-loaded _model/_tokenizer (fast path)."""
#     import numpy as np
#     with torch.inference_mode():
#         enc = _tokenizer(
#             smiles,
#             padding=True,
#             truncation=True,
#             max_length=_max_len,
#             return_tensors="pt",
#         )
#         input_ids = enc["input_ids"].to(_device)
#         attention_mask = enc.get("attention_mask")
#         if attention_mask is not None:
#             attention_mask = attention_mask.to(_device)

#         y = _model(input_ids=input_ids, attention_mask=attention_mask)  # shape (B, 1)
#         y = y.detach().cpu().numpy().reshape(-1, 1)

#         if _scaler is not None:
#             import joblib  # already imported in inference, but safe
#             y = _scaler.inverse_transform(y)

#     return [float(v) for v in y.flatten().tolist()]

# def _predict_via_subprocess(smiles: List[str]) -> List[float]:
#     """
#     Call: python inference.py --config CONFIG_PATH --smiles "S1" "S2" ...
#     Parse its stdout lines:  <SMILES>\t<VALUE>
#     """
#     cmd = [
#         sys.executable,
#         str(PKG_ROOT / "inference.py"),
        
#         "--config",
#         str(CONFIG_PATH),
#         "--smiles",
#         *smiles,
#     ]
#     proc = subprocess.run(cmd, capture_output=True, text=True)
#     if proc.returncode != 0:
#         raise RuntimeError(f"inference.py failed: {proc.stderr.strip()}")

#     preds = []
#     lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
#     # Each line: SMILES \t VALUE
#     for ln in lines:
#         parts = ln.split("\t")
#         if len(parts) == 2:
#             try:
#                 preds.append(float(parts[1]))
#             except ValueError:
#                 pass
#     if len(preds) != len(smiles):
#         raise RuntimeError(f"Parsed {len(preds)} predictions for {len(smiles)} SMILES.\nRaw:\n{proc.stdout}")
#     return preds
def _predict_inprocess_for_property(prop: str, smiles: List[str]) -> List[float]:
    """Use already-loaded head + tokenizer for a given property."""
    import numpy as np

    if prop not in _prop_models:
        raise ValueError(
            f"Property '{prop}' not loaded. Available: {list(_prop_models.keys())}"
        )

    model = _prop_models[prop]
    scaler = _prop_scalers.get(prop)
    tokenizer = _prop_tokenizers[prop]
    max_len = _prop_maxlens.get(prop, 256)

    with torch.inference_mode():
        enc = tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(_device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(_device)

        y = model(input_ids=input_ids, attention_mask=attention_mask)  # (B, 1)
        y = y.detach().cpu().numpy().reshape(-1, 1)

        if scaler is not None:
            y = scaler.inverse_transform(y)

    return [float(v) for v in y.flatten().tolist()]



def _predict_via_subprocess_for_property(prop: str, smiles: List[str]) -> List[float]:
    """
    Subprocess fallback: call inference.py with the correct config for this property.
    Assumes inference.py reads best_model_path from YAML.
    """
    cfg_path = PROPERTY_CONFIGS.get(prop)
    if cfg_path is None or not cfg_path.exists():
        raise ValueError(f"No config found for property '{prop}' (path={cfg_path})")

    cmd = [
        sys.executable,
        str(PKG_ROOT / "inference.py"),
        "--config",
        str(cfg_path),
        "--smiles",
        *smiles,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"inference.py failed: {proc.stderr.strip()}")

    preds = []
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) == 2:
            try:
                preds.append(float(parts[1]))
            except ValueError:
                pass

    if len(preds) != len(smiles):
        raise RuntimeError(
            f"Parsed {len(preds)} predictions for {len(smiles)} SMILES.\nRaw:\n{proc.stdout}"
        )

    return preds


async def _predict_property_tool(
    prop: str, smiles: list[str], ctx: Context
) -> list[float]:
    s_list = _safe_list_str(smiles)
    if not s_list:
        raise ValueError("Provide at least one SMILES string.")

    await ctx.info(f"Running {prop} inference on {len(s_list)} SMILES (device={_device})")

    with _lock:
        # 🔹 Ensure startup load has run in THIS process (mcp dev / mcp run)
        global _loaded_via_import
        if not _loaded_via_import:
            _startup_load()
            if not _loaded_via_import:
                # _startup_load failed; logs will show the real error
                raise RuntimeError("In-process models not ready; check MCP startup logs.")

        # 🔹 Lazy-build the model for this property
        if prop not in _prop_models:
            _build_property(prop)

        return _predict_inprocess_for_property(prop, s_list)



# @mcp.tool()
# async def predict_pei(smiles: list[str], ctx: Context) -> list[float]:
#     """
#     Predicts PE_I for one or more polymer SMILES strings using your finetuned TransPolymer downstream regressor.
#     Args:
#         smiles: list of SMILES strings (each a polymer sequence)
#     Returns:
#         List of predicted PE_I values (float), aligned to the input order.
#     """
#     s_list = _safe_list_str(smiles)
#     if not s_list:
#         raise ValueError("Provide at least one SMILES string.")

#     await ctx.info(f"Running PE_I inference on {len(s_list)} SMILES(device={_device})")
#     with _lock:
#         if _loaded_via_import:
#             return _predict_inprocess(s_list)
#         else:
#             return _predict_via_subprocess(s_list)
# @mcp.tool()       
# async def predict_opv(smiles: list[str], ctx: Context) -> list[float]:
#     s_list = _safe_list_str(smiles)     
#     if not s_list:
#         raise ValueError("Provide at least one SMILES string.")                                                                                                                     
                                                                    
#     await ctx.info(f"Running OPV inference on {len(s_list)} SMILES (device={_device})")                                                                                             
#     with _lock:                                                                                                                                                                     
#         if _loaded_via_import:                                                                                                                                                      
#             return _predict_inprocess(s_list)                                                                                                                                       
#         else:                                                                                                                                                                       
#             return _predict_via_subprocess(s_list) 
#     """                                                                                                                                                                             
#     Predicts OPV for one or more polymer SMILES strings using your finetuned TransPolymer downstream regressor.                                                                     
#     Args:                                                                                                                                                                           
#         smiles: list of SMILES strings (each a polymer sequence)                                                                                                                    
#     Returns:                                                                                                                                                                        
#         List of predicted OPV values (float), aligned to the input order.                                                                                                           
#     """                                                                                                                                                                             
 

# # Optional: echo tool for quick sanity checks
# @mcp.tool()
# async def ping(msg: str, ctx: Context) -> str:
#     """Simple echo to verify the MCP server is alive."""
#     await ctx.info("pong")
#     return f"pong: {msg}"
@mcp.tool()
async def predict_pei(smiles: list[str], ctx: Context) -> list[float]:
    """Predicts PE_I for one or more polymer SMILES strings."""
    return await _predict_property_tool("PE_I", smiles, ctx)


@mcp.tool()
async def predict_opv(smiles: list[str], ctx: Context) -> list[float]:
    """Predicts OPV for one or more polymer SMILES strings."""
    return await _predict_property_tool("OPV", smiles, ctx)


@mcp.tool()
async def predict_eea(smiles: list[str], ctx: Context) -> list[float]:
    """Predicts Eea for one or more polymer SMILES strings."""
    return await _predict_property_tool("Eea", smiles, ctx)


@mcp.tool()
async def predict_egb(smiles: list[str], ctx: Context) -> list[float]:
    """Predicts Egb for one or more polymer SMILES strings."""
    return await _predict_property_tool("Egb", smiles, ctx)


@mcp.tool()
async def predict_eps(smiles: list[str], ctx: Context) -> list[float]:
    """Predicts EPS for one or more polymer SMILES strings."""
    return await _predict_property_tool("EPS", smiles, ctx)

from typing import Optional, Dict, List

@mcp.tool()
async def predict_all_properties(smiles: list[str], ctx: Context) -> Dict[str, List[Optional[float]]]:
    """
    Predict Eea, Egb, EPS, PE_I and OPV for each SMILES.
    Returns:
        {
          "Eea": [...],
          "Egb": [...],
          "EPS": [...],
          "PE_I": [...],
          "OPV": [...]
        }
    """
    s_list = _safe_list_str(smiles)
    if not s_list:
        raise ValueError("Provide at least one SMILES string.")

    results: dict[str, list[float]] = {}
    for prop in PROPERTIES:
        try:
            preds = await _predict_property_tool(prop, s_list, ctx)
            results[prop] = preds
        except Exception as e:
            await ctx.info(f"{prop} prediction failed: {e}")
            results[prop] = [None] * len(s_list)
    return results



if __name__ == "__main__":
    _startup_load()
    mcp.run(transport="stdio")
