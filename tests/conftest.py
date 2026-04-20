"""Hypothesis profile registration."""

from hypothesis import HealthCheck, settings

settings.register_profile(
    "fuzz",
    deadline=None,
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci-fast",
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("fuzz")
