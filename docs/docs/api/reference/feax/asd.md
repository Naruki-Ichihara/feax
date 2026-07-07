---
sidebar_label: asd
title: feax.asd
---

Automatic Sparse Differentiation (ASD) utilities, built on `asdex`.

feax&#x27;s standard volume assembly does NOT go through this module: the
element-dense ``jacfwd`` + slot-map scatter is already optimal when the
sparsity is the mesh connectivity (a global coloring would need ``max row nnz``
colors — more kernel evaluations than the element dimension). This module
covers the places where a sparse operator is needed but its pattern is *not*
the plain connectivity:

* **extra residual terms** — arbitrary user coupling with unknown sparsity:
  :func:`detect_jacobian_pattern` / :func:`sparse_jacobian_fn` (used by the
  assembled ``extra_residual_fn`` solver path, enabling direct solvers).
* **reduced periodic operators** ``PᵀJP`` — pattern by boolean triple product
  (:func:``1), values by colored probes of the
  matrix-free action (:func:``2); enables direct/AMG solves
  for periodic problems.
* **design-space Hessians** ``d²J/dρ²`` — :func:``5 with
  symmetric (star) coloring + HVPs, for second-order design optimization.
* **verification** — :func:``6 checks feax&#x27;s hand-built
  CSR pattern against detection on the actual residual.

All factories return functions with a FIXED sparsity structure (jit-safe) that
produce :class:``7 — the operator type the feax solver stack
consumes.

#### detect\_jacobian\_pattern

```python
def detect_jacobian_pattern(f: Callable, x_sample) -> sp.csr_matrix
```

Global Jacobian sparsity of ``f`` (valid for all inputs) as boolean CSR.

Uses asdex&#x27;s jaxpr abstract interpretation — no derivative evaluation.

#### detect\_hessian\_pattern

```python
def detect_hessian_pattern(f: Callable, x_sample) -> sp.csr_matrix
```

Global Hessian sparsity of scalar ``f`` as boolean CSR.

#### connectivity\_pattern

```python
def connectivity_pattern(problem) -> sp.csr_matrix
```

feax&#x27;s assembled CSR pattern (mesh connectivity) as boolean CSR.

Requires ``MatrixView.FULL`` (UPPER/LOWER store a triangular view whose
pattern is not the full operator&#x27;s).

#### reduced\_operator\_pattern

```python
def reduced_operator_pattern(P, K_pattern) -> sp.csr_matrix
```

Sparsity of the Galerkin product ``PᵀKP`` by boolean triple product.

``P`` is the (n_full, n_reduced) prolongation (BCOO/scipy/dense);
``K_pattern`` any pattern accepted by this module (e.g.
:func:`connectivity_pattern`). Exact for boolean algebra — a superset of the
numerical pattern, which is what coloring/decompression need.

#### merge\_csr\_patterns

```python
def merge_csr_patterns(pattern_a, pattern_b) -> dict
```

Union of two CSR patterns plus the maps to assemble/transpose on it.

Returns a dict with the merged ``indptr``/``indices`` (int32 JAX arrays),
``nnz``, ``shape``, data-slot maps ``slots_a``/``slots_b`` (aligned with each
input pattern&#x27;s CSR order), and transpose maps ``T_perm``/``T_indptr``/
``T_indices`` for :func:``8.

#### sparse\_jacobian\_fn

```python
def sparse_jacobian_fn(f: Callable,
                       x_sample=None,
                       pattern=None,
                       *,
                       mode=None) -> Tuple[Callable, sp.csr_matrix]
```

Sparse Jacobian of ``f`` with a fixed structure.

Detects the sparsity from ``f``/``x_sample`` (or uses the given ``pattern``
superset), colors it, and returns ``(jac_fn, pattern_csr)`` where
``jac_fn(x) -&gt; CSRMatrix`` runs one JVP/VJP per color (jit-safe, fixed
structure). Cost per call: ``num_colors`` AD passes of ``f``.

#### sparse\_hessian\_fn

```python
def sparse_hessian_fn(f: Callable,
                      x_sample=None,
                      pattern=None,
                      *,
                      mode=None,
                      symmetric=True) -> Tuple[Callable, sp.csr_matrix]
```

Sparse Hessian of scalar ``f`` with a fixed structure.

Star (symmetric) coloring + one HVP per color. Returns
``(hess_fn, pattern_csr)`` with ``hess_fn(x) -&gt; CSRMatrix``. Intended e.g.
for design-space Hessians ``d²J/dρ²`` in second-order topology
optimization, where the pattern is the filter-stencil overlap.

#### operator\_assembler

```python
def operator_assembler(pattern, *, mode="fwd") -> Callable
```

Assembler for LINEAR operators known only through their matvec.

Colors ``pattern`` once; the returned ``assemble(matvec) -&gt; CSRMatrix``
materializes any linear operator with that sparsity using ``num_colors``
matvec probes (colored JVPs at 0). Used for the reduced periodic operator
``PᵀJP``, whose action exists matrix-free but whose assembled form is
needed for direct/AMG solves.

#### verify\_jacobian\_pattern

```python
def verify_jacobian_pattern(problem, traced_params, ts=None) -> dict
```

Check feax&#x27;s hand-built CSR pattern against detection on the residual.

Detects the true Jacobian sparsity of the assembled (bulk, no-BC) residual
via asdex and compares with :func:`connectivity_pattern`. Soundness requires
``detected ⊆ connectivity``; ``coverage`` reports how much of the
connectivity pattern is actually used (element blocks may hold structural
zeros). Returns ``dict(ok, num_detected, num_pattern, num_missing,
coverage)``.

