# Boundary-signal study, Metric A (PREREGISTRATION.md SS6): physical-event
# alignment. Positive step/boundary = within +/-2 steps of a labeled event.
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

EVENT_TYPES = ["wall_contact", "direction_turn", "corridor_change", "junction_arrival"]


def dilate_events(event_bool_60, radius=2):
    """event_bool_60: (60,) bool. Returns (60,) bool, True within +/-radius
    of any True in the input."""
    idx = np.where(event_bool_60)[0]
    if len(idx) == 0:
        return np.zeros(60, dtype=bool)
    out = np.zeros(60, dtype=bool)
    for i in idx:
        lo, hi = max(0, i - radius), min(60, i + radius + 1)
        out[lo:hi] = True
    return out


def step_level_auroc_ap(signal_60, events_dict, radius=2):
    """signal_60: (60,) continuous score. events_dict: {type: (60,) bool}.
    Positive = within +/-radius of ANY event type (pooled). Returns
    (auroc, ap) or (None, None) if degenerate (all-positive/all-negative)."""
    pooled = np.zeros(60, dtype=bool)
    for etype in EVENT_TYPES:
        pooled |= events_dict[etype]
    labels = dilate_events(pooled, radius)
    if labels.all() or not labels.any():
        return None, None
    try:
        auroc = roc_auc_score(labels, signal_60)
        ap = average_precision_score(labels, signal_60)
    except ValueError:
        return None, None
    return float(auroc), float(ap)


def boundary_level_prf1(boundaries, events_dict, radius=2, by_type=True):
    """boundaries: list of chosen boundary indices (0..59). events_dict:
    {type: (60,) bool} raw (undilated) event indicators -- dilation is
    applied here per-boundary/per-event, matching "within +/-2 steps".
    Returns dict: {'overall': {precision,recall,f1}, <type>: {...}, ...}."""
    boundaries = list(boundaries)
    result = {}

    def prf1_for(event_positions):
        if len(event_positions) == 0 and len(boundaries) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if len(boundaries) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        if len(event_positions) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        # precision: fraction of boundaries within radius of SOME event
        hit_b = sum(1 for b in boundaries if min(abs(b - e) for e in event_positions) <= radius)
        precision = hit_b / len(boundaries)
        # recall: fraction of events within radius of SOME boundary
        hit_e = sum(1 for e in event_positions if min(abs(b - e) for b in boundaries) <= radius)
        recall = hit_e / len(event_positions)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    pooled_positions = np.where(np.any([events_dict[t] for t in EVENT_TYPES], axis=0))[0].tolist()
    result["overall"] = prf1_for(pooled_positions)

    if by_type:
        for etype in EVENT_TYPES:
            positions = np.where(events_dict[etype])[0].tolist()
            result[etype] = prf1_for(positions)

    return result
