"""Graph-learning conjunction pre-screener.

Graph: nodes = catalogued objects, edges = pairs that pass the apogee/perigee filter.
Node embeddings are built with two rounds of GraphSAGE-style mean-aggregation message
passing over orbital-element features; each candidate edge is then scored by a learned
edge-readout classifier that also sees physics edge features (approximate MOID from the
mutual node line, relative inclination, altitude overlap).

The model is trained on yesterday's screening outcome (window [t-24h, t]) and predicts
which pairs need fine screening in the next window, so the expensive pairwise screening
only runs on a short list. KPIs: recall of real conjunctions and fraction of pairs pruned.

(Message passing is fixed/mean-aggregation and the readout is gradient boosting so the
prototype runs on any laptop with scikit-learn. Swapping in a trainable PyTorch-Geometric
GraphSAGE/GAT only replaces ``node_embeddings`` + the classifier.)
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from ..astro.propagate import RE, Catalog

TYPES = ["PAYLOAD", "ROCKET BODY", "DEBRIS", "UNKNOWN"]


def node_features(catalog: Catalog):
    rows = []
    for o in catalog.objects:
        i, raan, w = math.radians(o.inc_deg), math.radians(o.raan_deg), math.radians(o.argp_deg)
        rows.append([math.log(o.sma_km), o.ecc * 10, math.cos(i), math.sin(i), math.cos(raan),
                     math.sin(raan), math.cos(w), math.sin(w),
                     math.log10(abs(o.bstar) + 1e-8), math.log10((o.rcs_m2 or 0.1) + 1e-3),
                     (o.perigee_km) / 1000.0, (o.apogee_km) / 1000.0]
                    + [1.0 if o.obj_type == t else 0.0 for t in TYPES])
    return np.array(rows, dtype=np.float32)


def node_embeddings(catalog: Catalog, edges: np.ndarray, layers=2):
    X = node_features(catalog)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    N = len(X)
    if len(edges):
        A = sparse.coo_matrix((np.ones(len(edges) * 2, np.float32),
                               (np.concatenate([edges[:, 0], edges[:, 1]]),
                                np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(N, N)).tocsr()
    else:
        A = sparse.csr_matrix((N, N), dtype=np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    Dinv = sparse.diags(1.0 / np.maximum(deg, 1)).astype(np.float32)
    H = X
    for _ in range(layers):
        H = np.hstack([H, Dinv @ (A @ H)])          # GraphSAGE-mean aggregation
    H = np.hstack([H, np.log1p(deg)[:, None]])        # local traffic density
    return H.astype(np.float32)


def _geometry(catalog: Catalog):
    P, Q, W, a, e, per, apo = [], [], [], [], [], [], []
    for o in catalog.objects:
        i, O, w = math.radians(o.inc_deg), math.radians(o.raan_deg), math.radians(o.argp_deg)
        cO, sO, ci, si, cw, sw = math.cos(O), math.sin(O), math.cos(i), math.sin(i), math.cos(w), math.sin(w)
        P.append([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si])
        Q.append([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si])
        W.append([sO * si, -cO * si, ci])
        a.append(o.sma_km); e.append(o.ecc); per.append(o.perigee_km); apo.append(o.apogee_km)
    return (np.array(P), np.array(Q), np.array(W), np.array(a), np.array(e), np.array(per),
            np.array(apo))


def edge_features(H, geo, edges):
    P, Q, W, a, e, per, apo = geo
    u, v = edges[:, 0], edges[:, 1]
    k = np.cross(W[u], W[v])
    sk = np.linalg.norm(k, axis=1)
    k = k / np.maximum(sk, 1e-9)[:, None]

    def r_at(idx, d):
        nu = np.arctan2(np.einsum("ij,ij->i", d, Q[idx]), np.einsum("ij,ij->i", d, P[idx]))
        return a[idx] * (1 - e[idx] ** 2) / (1 + e[idx] * np.cos(nu))

    d1 = np.abs(r_at(u, k) - r_at(v, k))
    d2 = np.abs(r_at(u, -k) - r_at(v, -k))
    moid = np.minimum(d1, d2)
    coplanar = sk < 0.02
    moid = np.where(coplanar, np.maximum(np.maximum(per[u], per[v]) - np.minimum(apo[u], apo[v]), 0), moid)
    rel_inc = np.degrees(np.arcsin(np.clip(sk, 0, 1)))
    overlap = np.minimum(apo[u], apo[v]) - np.maximum(per[u], per[v])
    g = np.column_stack([np.log1p(moid), rel_inc, np.log1p(np.maximum(overlap, 0)),
                         np.abs(a[u] - a[v]), e[u] + e[v], np.minimum(per[u], per[v]) / 1000])
    return np.hstack([np.abs(H[u] - H[v]), (H[u] + H[v]) * 0.5, g.astype(np.float32)]), moid


class GNNScreener:
    def __init__(self, seed=0):
        self.seed = seed
        self.clf = None
        self.threshold = 0.5
        self.metrics = {}

    def fit(self, catalog, edges, positive_pairs: set, target_recall=0.99, neg_ratio=40):
        rng = np.random.default_rng(self.seed)
        H = node_embeddings(catalog, edges)
        geo = _geometry(catalog)
        y = np.array([(int(i), int(j)) in positive_pairs for i, j in edges])
        pos = np.nonzero(y)[0]; neg = np.nonzero(~y)[0]
        if len(pos) < 5:
            self.metrics = {"status": "not enough positive pairs to train", "positives": int(len(pos))}
            return self
        neg = rng.choice(neg, min(len(neg), max(len(pos) * neg_ratio, 20000)), replace=False)
        idx = np.concatenate([pos, neg]); rng.shuffle(idx)
        X, _ = edge_features(H, geo, edges[idx])
        yy = y[idx]
        cut = int(0.8 * len(idx))
        t0 = time.perf_counter()
        self.clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_bins=64,
                                                  class_weight="balanced", random_state=self.seed)
        self.clf.fit(X[:cut], yy[:cut])
        pv = self.clf.predict_proba(X[cut:])[:, 1]
        pos_scores = np.sort(pv[yy[cut:]])
        if len(pos_scores):
            self.threshold = float(pos_scores[int(math.floor((1 - target_recall) * len(pos_scores)))]) * 0.98
        self.clf.fit(X, yy)   # refit on everything
        self.metrics = {"train_edges": int(len(idx)), "train_positives": int(len(pos)),
                        "train_s": round(time.perf_counter() - t0, 2),
                        "val_auc": float(roc_auc_score(yy[cut:], pv)) if yy[cut:].any() else None,
                        "threshold": self.threshold}
        return self

    def score(self, catalog, edges, batch=200_000):
        H = node_embeddings(catalog, edges)
        geo = _geometry(catalog)
        out = np.zeros(len(edges), np.float32)
        moids = np.zeros(len(edges), np.float32)
        for k in range(0, len(edges), batch):
            X, m = edge_features(H, geo, edges[k:k + batch])
            out[k:k + batch] = self.clf.predict_proba(X)[:, 1] if self.clf else 1.0
            moids[k:k + batch] = m
        return out, moids

    def evaluate(self, catalog, edges, positive_pairs: set):
        s, moid = self.score(catalog, edges)
        y = np.array([(int(i), int(j)) in positive_pairs for i, j in edges])
        keep = s >= self.threshold
        res = {
            "edges": int(len(edges)), "positives": int(y.sum()),
            "test_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None,
            "test_avg_precision": float(average_precision_score(y, s)) if y.any() else None,
            "recall": float((keep & y).sum() / max(y.sum(), 1)),
            "kept_edges": int(keep.sum()),
            "pruned_fraction": float(1 - keep.mean()) if len(keep) else 0.0,
        }
        # physics-only baseline at the same recall: MOID filter
        if y.any():
            thr_m = np.sort(moid[y])[min(len(moid[y]) - 1, int(math.ceil(res["recall"] * y.sum())) - 1)]
            res["moid_baseline_pruned_fraction_same_recall"] = float(1 - (moid <= thr_m).mean())
        self.metrics.update({"test_" + k if not k.startswith("test") else k: v for k, v in res.items()})
        return keep, s
