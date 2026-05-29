"""Lightweight logging utilities for CorVer reward scoring.

Self-contained — no external config files or project dependencies required.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_timestamp_singleton = None


def get_timestamp() -> str:
    """Return current timestamp as 'YYYY-MM-DD HH:MM:SS'."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_output_timestamp() -> str:
    """Return a singleton timestamp for consistent directory naming within a session."""
    global _timestamp_singleton
    if _timestamp_singleton is None:
        _timestamp_singleton = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _timestamp_singleton


def save_json_output(
    data: Dict[str, Any],
    function_name: str,
    output_dir: str,
    sample_index: Optional[int] = None,
) -> str:
    """Save scoring results to a JSON file with merge-append semantics.

    Args:
        data: Dictionary of results to save.
        function_name: Base name for the output file.
        output_dir: Directory to write to. Created if it does not exist.
        sample_index: If given, appended to the filename.

    Returns:
        Path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if sample_index is not None:
        filename = f"{function_name}_sample_{sample_index}.json"
    else:
        filename = f"{function_name}.json"

    filepath = os.path.join(output_dir, filename)

    existing_data = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Cannot parse existing JSON file: {filepath}, will create new file")

    if isinstance(data, dict) and isinstance(existing_data, dict):
        for key, value in data.items():
            if key in existing_data and isinstance(value, list) and isinstance(existing_data[key], list):
                existing_data[key].extend(value)
            else:
                existing_data[key] = value
        merged_data = existing_data
    else:
        merged_data = data

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    return filepath
