"""Multi-objective hyper-parameter / threshold optimisation scaffold (numpy only).

The optimiser is a plug-in: anything with  ask() -> dict  and  tell(params, score).
RandomSearch and a small evolution strategy are included. Put GDOA (or any other
metaheuristic) behind the same two methods; it is deliberately NOT the main novelty.

Searchable: learning rate, loss weights, augmentation strength, attention/graph
settings, detection threshold, candidate threshold, FP-suppression thresholds.
"""
import math

import numpy as np

DEFAULT_SPACE = {
    "lr": ("log", 1e-5, 1e-3),
    "w_det": ("lin", 0.5, 2.0), "w_slice": ("lin", 0.2, 2.0), "w_ves": ("lin", 0.1, 1.0),
    "w_cl": ("lin", 0.0, 1.0), "w_node": ("lin", 0.1, 1.0), "w_q": ("lin", 0.1, 1.0), "w_sev": ("lin", 0.1, 1.0),
    "aug_noise": ("lin", 0.0, 0.2), "graph_layers": ("int", 1, 4),
    "det_thr": ("lin", 0.2, 0.8), "cand_thr": ("lin", 0.3, 0.8), "fp_min_vessel_overlap": ("lin", 0.4, 0.95),
}


def sample(space, rng):
    p = {}
    for k, (kind, lo, hi) in space.items():
        if kind == "log":
            p[k] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        elif kind == "int":
            p[k] = int(rng.integers(lo, hi + 1))
        else:
            p[k] = float(rng.uniform(lo, hi))
    return p


def clip(space, p):
    for k, (kind, lo, hi) in space.items():
        p[k] = int(round(min(max(p[k], lo), hi))) if kind == "int" else float(min(max(p[k], lo), hi))
    return p


def composite_objective(m, lam=None):
    """Lower is better. Jointly penalises classification error, localisation error, false
    positives, quantification error and calibration error (not only accuracy).

    m: dict with auc_det, lesion_sens, fp_per_scan, dice (or slice_auc), bi_mae, ece
    """
    lam = lam or dict(auc=1.0, sens=1.0, fp=0.3, loc=0.5, bi=0.5, ece=0.3)
    return (lam["auc"] * (1 - m["auc_det"]) + lam["sens"] * (1 - m["lesion_sens"])
            + lam["fp"] * m["fp_per_scan"] + lam["loc"] * (1 - m["dice"])
            + lam["bi"] * m.get("bi_mae", 0.0) + lam["ece"] * m.get("ece", 0.0))


class RandomSearch:
    def __init__(self, space=None, seed=0):
        self.space, self.rng, self.hist = space or DEFAULT_SPACE, np.random.default_rng(seed), []

    def ask(self):
        return sample(self.space, self.rng)

    def tell(self, params, score):
        self.hist.append((params, score))


class SimpleES:
    """(mu + lambda) evolution strategy with Gaussian mutation; stand-in metaheuristic."""

    def __init__(self, space=None, mu=4, sigma=0.15, seed=0):
        self.space, self.mu, self.sigma = space or DEFAULT_SPACE, mu, sigma
        self.rng, self.pop = np.random.default_rng(seed), []

    def ask(self):
        if len(self.pop) < self.mu:
            return sample(self.space, self.rng)
        parent = self.pop[int(self.rng.integers(len(self.pop)))][0]
        child = {}
        for k, (kind, lo, hi) in self.space.items():
            if kind == "log":
                child[k] = math.exp(math.log(parent[k]) + self.rng.normal(0, self.sigma) * (math.log(hi) - math.log(lo)))
            else:
                child[k] = parent[k] + self.rng.normal(0, self.sigma) * (hi - lo)
        return clip(self.space, child)

    def tell(self, params, score):
        self.pop.append((params, score))
        self.pop = sorted(self.pop, key=lambda t: t[1])[: self.mu]


def optimize(train_eval_fn, optimizer, budget=20):
    """train_eval_fn(params) -> metrics dict (see composite_objective). Returns best (params, score, metrics)."""
    best = (None, float("inf"), None)
    for _ in range(budget):
        p = optimizer.ask()
        met = train_eval_fn(p)
        s = composite_objective(met)
        optimizer.tell(p, s)
        if s < best[1]:
            best = (p, s, met)
    return best
