"""Structured result type for `check_*` invariant predicates."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict


class InvariantReport(BaseModel):
    """Outcome of a single invariant check (ADR-0001 §(ii)).

    `check_*` predicates return one of these on every invariant outcome
    (pass or violation). The structured form lets the visualizer (P5+)
    render outcomes without parsing exception strings. ``check_*`` may
    still raise ``AssertionError`` on programmer errors (calling a check
    that needs a modality the sample doesn't carry).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    """Stable identifier of the form ``"<modality>.<invariant>"``."""

    ok: bool
    """``True`` iff the invariant holds on the supplied inputs."""

    message: str | None = None
    """Human-readable reason when ``ok`` is ``False``; ``None`` otherwise."""

    details: Mapping[str, int | float | str] | None = None
    """Optional structured key/value context for the visualizer."""


def raise_if_violated(report: InvariantReport) -> None:
    """Trampoline used by every ``assert_*`` to lift a report into a raise."""
    if not report.ok:
        raise AssertionError(report.message or report.name)
