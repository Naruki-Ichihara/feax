"""Ordering a batch of designs by similarity (for grouped batched solves).

A shared preconditioner (:func:`feax.shared_factor`) is only as good as the
designs of its group are alike, so a batch of designs is best solved in groups
of *similar* ones. These helpers order a batch so that consecutive designs
are similar; the solver's ``chunk=`` then groups them. They work on any
feature vector per design, ``(B, f)``: the design field itself, the SIMP
modulus (on a log scale), or — independent of the parametrization — the
operator signature of the batched solver
(:meth:`feax.BatchedSolver.operator_signature`, the log-diagonal of every
stiffness matrix).
"""

import jax
import jax.numpy as np


def _center(features, weights):
    F = np.asarray(features)
    if F.ndim != 2:
        raise ValueError(f"features must be (B, f), got shape {F.shape}.")
    w = None if weights is None else np.asarray(weights)
    Fc = F - np.mean(F, axis=0, keepdims=True)
    if w is not None:
        Fc = Fc * np.sqrt(w)[None, :]
    return Fc


def similarity_order(features, weights=None):
    """Order the designs along their principal direction.

    Projects the (centered, optionally weighted) feature vectors on the first
    principal axis of the batch and returns the permutation that sorts the
    projections, so that consecutive designs are close and a contiguous group
    of ``chunk`` designs is a cluster along the batch's main direction. Exact
    when the designs lie along one path (a parameter sweep, load angles);
    a good first cut otherwise. ``O(B² f)``.

    Parameters
    ----------
    features : array (B, f)
        One feature vector per design.
    weights : array (f,), optional
        Non-negative weights of the features (element volumes, ...).

    Returns
    -------
    order : int array (B,)
        ``features[order]`` is the similarity-sorted batch.
    """
    Fc = _center(features, weights)
    gram = Fc @ Fc.T                                            # (B, B)
    _, vecs = np.linalg.eigh(gram)
    proj = vecs[:, -1]                                          # principal scores (up to sign)
    return np.argsort(proj)


def group_spread(features, chunk, weights=None, order=None):
    """Within-group spread, the quantity a shared preconditioner suffers from.

    Groups the designs (in ``order``, or as given) in consecutive groups of
    ``chunk`` and returns, per design, the weighted distance between its
    feature vector and the mean of its group, ``‖F_i − F_G‖_w``; the batch
    mean of that is a one-number diagnostic to compare orderings.
    """
    F = np.asarray(features)
    B = F.shape[0]
    if order is not None:
        F = F[np.asarray(order)]
    w = np.ones(F.shape[1]) if weights is None else np.asarray(weights)
    C = int(chunk) if chunk else B
    G = -(-B // C)
    idx = np.minimum(np.arange(G * C), B - 1)
    Fg = F[idx].reshape(G, C, -1)
    dev = Fg - np.mean(Fg, axis=1, keepdims=True)
    d = np.sqrt(np.sum(dev ** 2 * w, axis=-1)).reshape(-1)[:B]
    if order is not None:
        d = d.at[np.asarray(order)].set(d)
    return d


__all__ = ["similarity_order", "group_spread"]
