# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for file I/O with yaml."""

import os
import yaml
import torch
from typing import Any


def class_to_dict(obj: object) -> dict[str, Any]:
    """Recursively convert a class instance or class itself into a dictionary, handling nested classes and lists correctly."""
    result = {}

    # If it's a class, instantiate it
    if isinstance(obj, type):
        obj = obj()

    attr_names = [
        name for name in dir(obj)
        if not name.startswith("__") and not callable(getattr(obj, name, None))
    ]

    for name in attr_names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue

        # ✨ Base types: keep as-is
        if isinstance(value, (int, float, str, bool, type(None))):
            result[name] = value

        # ✨ Tensor: convert to list
        elif isinstance(value, torch.Tensor):
            result[name] = value.tolist()

        # ✨ List or tuple: handle item by item
        elif isinstance(value, (list, tuple)):
            result[name] = [
                class_to_dict(v) if isinstance(v, (dict, type)) or hasattr(v, "__dict__") else v
                for v in value
            ]

        # ✨ Dict: convert keys/values if necessary
        elif isinstance(value, dict):
            result[name] = {
                k: class_to_dict(v) if isinstance(v, (dict, type)) or hasattr(v, "__dict__") else v
                for k, v in value.items()
            }

        # ✨ Nested class (still a type)
        elif isinstance(value, type):
            result[name] = class_to_dict(value())

        # ✨ Instance of a class
        elif hasattr(value, "__dict__"):
            result[name] = class_to_dict(value)

        # ✨ Fallback: just keep raw value
        else:
            result[name] = value

    return result


def load_yaml(filename: str) -> dict:
    """Loads an input PKL file safely.

    Args:
        filename: The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename) as f:
        data = yaml.full_load(f)
    return data


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save either a dictionary or class object.
        sort_keys: Whether to sort the keys in the output file. Defaults to False.
    """
    # check ending
    if not filename.endswith("yaml"):
        filename += ".yaml"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # convert data into dictionary
    if not isinstance(data, dict):
        data = class_to_dict(data)
    # save data
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)
