"""Portable execution infrastructure; importing this package never runs a model."""


class PortabilityError(RuntimeError):
    """Fail closed when a path, environment, authorization or state is ambiguous."""
