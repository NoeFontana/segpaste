"""Hypothesis strategies for dense-sample tests."""

from tests.strategies.dense_sample import (
    MAX_FUZZ_SIZE,
    dense_sample_strategy,
    depth_fields_strategy,
    image_strategy,
    instance_fields_strategy,
    normals_strategy,
    panoptic_map_strategy,
    semantic_map_strategy,
)

__all__ = [
    "MAX_FUZZ_SIZE",
    "dense_sample_strategy",
    "depth_fields_strategy",
    "image_strategy",
    "instance_fields_strategy",
    "normals_strategy",
    "panoptic_map_strategy",
    "semantic_map_strategy",
]
