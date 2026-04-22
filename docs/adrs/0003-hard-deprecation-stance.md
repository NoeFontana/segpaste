# ADR-0003 — Hard deprecation stance for pre-1.0 breaks

|            |                                                                 |
| ---------- | --------------------------------------------------------------- |
| Number     | 0003                                                            |
| Title      | Hard deprecation stance (no experimental namespace, no runtime shims) |
| Status     | Accepted                                                        |
| Author     | @NoeFontana                                                     |
| Created    | 2026-04-21                                                      |
| Updated    | 2026-04-21                                                      |
| Tag        | `ADR-0003`                                                      |
| Supersedes | ADR-0001 Part (i) — deprecation policy clauses only             |

## Context

ADR-0001 Part (i) set a soft-deprecation policy for the 0.9.x stability line:
`CopyPasteTransform` was to move to a new `segpaste.experimental` sub-package
with a top-level re-export that emits `DeprecationWarning`; `DetectionTarget`
was to remain as a runtime-compatible subclass shim forwarding to
`DenseSample`. Both symbols were scheduled for removal in 0.10.0.

Since that ADR landed, two observations changed the calculus:

1. **The project is pre-1.0 and solo + agentic.** The SemVer disclaimer in
   `pyproject.toml` and `README.md` already warns that breaking changes may
   land without notice below 1.0.0. Soft-deprecation infrastructure
   (`segpaste.experimental`, runtime `DeprecationWarning` shims, subclass
   forwarders) buys nothing for a user base of zero and costs a full 0.9.x
   test-matrix entry plus a migration obligation that has to be carried on
   every refactor in P1.
2. **P1 (dense-sample composites) churns every transform.** Keeping a
   runtime-compatible `DetectionTarget` shim means every internal refactor
   must either preserve the subclass contract or route carefully around it.
   That is the opposite of what pre-1.0 is for.

The pre-1.0 free-break window closes at 1.0.0. Using it to actually remove
code is cheaper than using it to build more scaffolding.

## Decision

This ADR **supersedes only the deprecation policy clauses** of ADR-0001 Part
(i). The rest of ADR-0001 — invariants (Part ii), type-system decisions
(Part iii), seed and replay policy (Part iv), the `__version__` requirement,
the `_internal` namespace, the `blend_mode` tightening, and the ambiguous
`integrations` exports resolution — remains in force.

### `CopyPasteTransform` — delete outright in 0.9.0

- No `segpaste.experimental` sub-package is created.
- No top-level re-export with `DeprecationWarning`.
- The symbol, its module-level definition, and every reference in docs,
  README, CLAUDE.md, and the example scripts are removed in one commit.
- Consumers who need the old behavior pin `segpaste<0.9` on PyPI. That is
  the documented migration path and it is sufficient.
- Removal is logged under `### Removed` in `CHANGELOG.md` for the 0.9.0
  release.

### `DetectionTarget` — close the public surface in 0.9.0, remove the type in 0.9.1

- `DetectionTarget` is removed from `segpaste.__all__` and
  `segpaste.types.__all__` in 0.9.0. `segpaste.DetectionTarget` becomes an
  `AttributeError`.
- The class itself remains importable via its full internal path during
  0.9.0 because P1's W1 workstream still threads it through the augmentation
  pipeline. This is **not** a runtime-compatible shim: no subclass
  relationship with `DenseSample`, no `DeprecationWarning` on construction,
  no `__getattr__` magic on `segpaste` or `segpaste.types`.
- Conversion between the two types uses the bidirectional static methods
  already present on `DenseSample`: `DenseSample.from_detection_target(...)`
  and `DenseSample.to_detection_target(...)`. No new classmethod is added.
- An `xfail(strict=True)` test asserts that the class is gone from
  `segpaste.types` by 0.9.1. P1's W1 completes the type-level migration,
  deletes the class, and flips the test green.

### `segpaste.experimental` — not created

ADR-0001 Part (i)'s `segpaste.experimental` sub-package is not created.
The rationale for it was `CopyPasteTransform`; with `CopyPasteTransform`
deleted there is no inaugural resident. If a future unstable surface needs
to be exposed, a new ADR introduces the namespace then.

## Consequences

- **Breaking change on upgrade from 0.8.x to 0.9.0.** `segpaste.CopyPasteTransform`
  and `segpaste.DetectionTarget` both disappear from the top level. Memorialized
  in the 0.9.0 `CHANGELOG.md` and in the existing pre-1.0 SemVer disclaimer.
- **Simpler P1.** No experimental namespace to maintain, no subclass-shim
  contract to preserve while refactoring transforms.
- **Surface-lock enforcement becomes load-bearing.** The `__all__` match
  tests introduced in P0.E step (iii) are the forcing function that prevents
  the surface from re-accreting during P1 churn. Adding a public name now
  requires an ADR amendment.

## Status and supersession

- **Accepted** when this file lands on `main` with `Status: Accepted`.
- ADR-0001's header is updated in the same commit to `Status: Superseded in
  part by ADR-0003` with a scope note pointing back here ("Part (i)
  deprecation policy only — invariants, types, and seed policy remain in
  force").
- Any later decision that reintroduces soft-deprecation for pre-1.0 removals
  requires a new ADR that explicitly supersedes this one.

## Verification

- `uv run mkdocs build --strict` passes with this ADR in the nav.
- ADR-0001 renders with the updated `Status` cross-linking this ADR.
