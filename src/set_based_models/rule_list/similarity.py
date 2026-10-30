"""Event-level Jaccard matrices for fixed candidate rules."""

import numpy as np


_POPCOUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def event_jaccard_matrix(masks):
    """Calculate unweighted similarities; two empty selections have Jaccard 1."""
    masks = list(masks)
    size = len(masks)
    matrix = np.eye(size, dtype=np.float64)
    if size:
        if any(mask is None for mask in masks):
            raise RuntimeError("Event masks are missing; call recalculate_candidates() to rebuild similarities")
        if len({len(mask) for mask in masks}) != 1:
            raise ValueError("Candidate event masks must describe the same event rows")
        packed = np.stack([np.packbits(mask) for mask in masks])
        counts = np.array([np.count_nonzero(mask) for mask in masks], dtype=np.int64)
        for row in range(size):
            for start in range(row + 1, size, 128):
                end = min(start + 128, size)
                intersection = _POPCOUNT[packed[start:end] & packed[row]].sum(axis=1, dtype=np.int64)
                union = counts[start:end] + counts[row] - intersection
                values = np.divide(intersection, union, out=np.ones(end - start), where=union != 0)
                matrix[row, start:end] = matrix[start:end, row] = values
    matrix.flags.writeable = False
    return matrix


def rule_jaccard_matrices(rules):
    rules = list(rules)
    return (event_jaccard_matrix(rule.signal_selection for rule in rules),
            event_jaccard_matrix(rule.background_selection for rule in rules))


def merged_jaccard_matrices(models):
    """Preserve input blocks and assume zero similarity between input models.

    Old mask-free artifacts may lack matrices. In that case the merged cache
    stays unavailable until recalculation, rather than inventing within-block
    similarities. Sources are never modified.
    """
    pairs, blocks = [], []
    for model in models:
        pair = model.signal_jaccard_, model.background_jaccard_
        if any(matrix is None for matrix in pair):
            if any(rule.signal_selection is None or rule.background_selection is None
                   for rule in model.candidate_rules_):
                return None, None, ()
            pair = rule_jaccard_matrices(model.candidate_rules_)
        pairs.append(pair)
        blocks.extend(model.jaccard_blocks_ if model.jaccard_source_ == "block_diagonal"
                      else (len(model.candidate_rules_),))
    size = sum(len(model.candidate_rules_) for model in models)
    matrices = []
    for index in range(2):
        matrix = np.zeros((size, size), dtype=np.float64)
        offset = 0
        for pair in pairs:
            block = pair[index]
            end = offset + len(block)
            matrix[offset:end, offset:end] = block
            offset = end
        matrix.flags.writeable = False
        matrices.append(matrix)
    return matrices[0], matrices[1], tuple(blocks)


def jaccard_record(model):
    if model.signal_jaccard_ is None or model.background_jaccard_ is None:
        return None
    return {"metric": "jaccard", "scope": "event", "source": model.jaccard_source_,
            "statistics_source": model.statistics_source_, "rule_ids": list(range(len(model.candidate_rules_))),
            "blocks": list(model.jaccard_blocks_), "signal": model.signal_jaccard_.tolist(),
            "background": model.background_jaccard_.tolist()}


def restore_jaccard(model, record):
    size = len(model.candidate_rules_)
    if (record["metric"] != "jaccard" or record["scope"] != "event"
            or record["rule_ids"] != list(range(size)) or sum(record["blocks"]) != size):
        raise ValueError("Jaccard matrices must describe the model's event-level candidate rule order")
    matrices = []
    for role in ("signal", "background"):
        matrix = np.asarray(record[role], dtype=np.float64)
        if size == 0 and matrix.size == 0:
            matrix = matrix.reshape(0, 0)
        if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
            raise ValueError("Jaccard matrices must be square candidate matrices with finite values in [0, 1]")
        matrix.flags.writeable = False
        matrices.append(matrix)
    model.signal_jaccard_, model.background_jaccard_ = matrices
    model.jaccard_source_ = record["source"]
    model.jaccard_blocks_ = tuple(record["blocks"])
