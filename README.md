# PolyAgent MCP Server

An MCP server for polymer inverse design and property optimization.

## Features
- Single-property optimization
- Multi-property constrained optimization
- Latent-space decoding
- Chemistry-aware constraints
## Installation

This MCP server depends on two local polymer libraries and must be installed in **editable mode**.

---

### 1. Clone required repositories

```bash
git clone https://github.com/BARATI-LAB/transpolymer-polyagent.git
git clone https://github.com/BARATI-LAB/open_macromolecule-polyagent.git

pip install -e ./transpolymer-polyagent
pip install -e ./open_macromolecule-polyagent
```

### 2. Clone PolyAgent
```bash
git clone https://github.com/BARATI-LAB/polyagent.git
```

### 3. Create uv environment: https://docs.astral.sh/uv/getting-started/installation/ 
```bash
uv sync
```

### 4. Update the paths according your folders
```bash
gemini
```


