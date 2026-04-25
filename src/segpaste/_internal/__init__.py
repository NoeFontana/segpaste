"""Private surface for segpaste.

Anything not re-exported from `segpaste.__all__` is internal. W2 lands
``composite``, ``placement``, ``instance_paste``, and ``classmix`` here;
W3 adds ``panoptic_paste``; W4 adds ``depth_paste``. Promotion to the
public surface is deferred (ADR-0005 §5).

See ADR-0001 Part (i), ADR-0003, ADR-0005, ADR-0006, and ADR-0007.
"""
