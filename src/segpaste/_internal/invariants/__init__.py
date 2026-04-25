"""Per-modality invariants for ADR-0001 §(ii).

Each modality submodule exposes paired predicates: ``check_*`` returns
a :class:`InvariantReport` describing the invariant outcome; ``assert_*``
trampolines through ``check_*`` and raises ``AssertionError`` on
violation. Both forms raise ``AssertionError`` on programmer errors
(missing modality fields the check requires). The visualizer (P5+)
consumes the structured form; the test suite consumes either.
"""

from __future__ import annotations

from segpaste._internal.invariants._report import InvariantReport

__all__ = ["InvariantReport"]
