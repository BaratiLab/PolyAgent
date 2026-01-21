# tools/validate_constraints.py

from typing import Dict, Tuple, List

def validate_constraints_tool(
    property_names: List[str],
    target_property: str,
    constraints: Dict[str, Tuple[float, float]]
) -> Dict[str, str]:
    """
    MCP Tool: Validate constraint specification.
    """

    if target_property not in property_names:
        raise ValueError(f"Target property '{target_property}' not in model properties")

    for prop, (low, high) in constraints.items():
        if prop not in property_names:
            raise ValueError(f"Constraint property '{prop}' not in model properties")
        if low >= high:
            raise ValueError(f"Invalid range for {prop}: low >= high")

    return {"status": "valid"}
