from __future__ import annotations

import json
from typing import Any


def normalize_component_payload(value: Any) -> dict[str, Any] | None:
    """Accept Streamlit JSON objects and legacy JSON-string component values."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None

    if isinstance(value, dict) and isinstance(value.get("value"), str):
        # Defensive support for older components that wrapped JSON in `value`.
        try:
            nested = json.loads(value["value"])
            if isinstance(nested, dict):
                value = nested
        except json.JSONDecodeError:
            pass

    return value if isinstance(value, dict) else None
