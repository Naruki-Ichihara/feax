---
sidebar_label: similarity
title: feax.gene.similarity
---

Ordering a batch of designs by similarity (for grouped batched solves).

A shared preconditioner (:func:`feax.shared_factor`) is only as good as the
designs of its group are alike, so a batch of designs is best solved in groups
of *similar* ones. These helpers order a batch so that consecutive designs
are similar; the solver&#x27;s ``chunk=`` then groups them. They work on any
feature vector per design, ``(B, f)``: the design field itself, the SIMP
modulus (on a log scale), or — independent of the parametrization — the
operator signature of the batched solver
(:meth:`feax.BatchedSolver.operator_signature`, the log-diagonal of every
stiffness matrix).

#### similarity\_order

```python
def similarity_order(features, weights=None)
```

Order the designs along their principal direction.

Projects the (centered, optionally weighted) feature vectors on the first
principal axis of the batch and returns the permutation that sorts the
projections, so that consecutive designs are close and a contiguous group
of ``chunk`` designs is a cluster along the batch&#x27;s main direction. Exact
when the designs lie along one path (a parameter sweep, load angles);
a good first cut otherwise. ``O(B² f)``.

Parameters
----------
- **features** (*array (B, f)*): One feature vector per design.
- **weights** (*array (f,), optional*): Non-negative weights of the features (element volumes, ...).


Returns
-------
- **order** (*int array (B,)*): ``features[order]`` is the similarity-sorted batch.


#### group\_spread

```python
def group_spread(features, chunk, weights=None, order=None)
```

Within-group spread, the quantity a shared preconditioner suffers from.

Groups the designs (in ``order``, or as given) in consecutive groups of
``chunk`` and returns, per design, the weighted distance between its
feature vector and the mean of its group, ``‖F_i − F_G‖_w``; the batch
mean of that is a one-number diagnostic to compare orderings.

