import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "constraints.json"

with open(SCHEMA_PATH) as f:
    PROPERTY_SCHEMA = json.load(f)


def build_constraints(
    target_property: str,
    target_value: float,
    tolerance: float,
    include: list[str] | None = None,
    mode: str = "std",  # "std" | "pareto"
    std_factor: float = 1.0
) -> dict:
    """
    Generate constraints automatically from schema.
    """

    if target_property not in PROPERTY_SCHEMA:
        raise ValueError(f"Unknown target property: {target_property}")

    constraints = {}

    for prop, stats in PROPERTY_SCHEMA.items():
        if include and prop not in include:
            continue

        if prop == target_property:
            continue

        if mode == "std":
            mean = stats["mean"]
            std = stats["std"]
            constraints[prop] = [
                mean - std_factor * std,
                mean + std_factor * std
            ]

        elif mode == "pareto":
            constraints[prop] = [
                stats["median"],
                stats["pareto_80"]
            ]

        else:
            raise ValueError(f"Unknown constraint mode: {mode}")

    return {
        "target_property": target_property,
        "target_value": target_value,
        "tolerance": tolerance,
        "constraints": constraints
    }
