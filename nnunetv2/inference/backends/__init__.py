"""Backend-specific inference implementations.

Each subpackage exposes a ``ModelBundle`` (filesystem + config) and an
``InferenceEngine`` (pure compute) with the same surface, so callers can
target either backend through a structural ``Protocol``.
"""
