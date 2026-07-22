import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

from utils import ensure_dir, hash_cfg


EPS_TIME = 1e-9
_STATE_FOREST_WORKER = {}


def row_normalize(mat: sp.spmatrix) -> sp.csr_matrix:
    mat = mat.tocsr().astype(np.float32)
    rowsum = np.asarray(mat.sum(axis=1)).ravel()
    inv = np.zeros_like(rowsum, dtype=np.float32)
    mask = rowsum > 0
    inv[mask] = 1.0 / rowsum[mask]
    return sp.diags(inv).dot(mat).tocsr()


def sparse_row_topk(mat: sp.spmatrix, k: int) -> sp.csr_matrix:
    mat = mat.tocsr()
    if k <= 0:
        return mat
    rows, cols, data = [], [], []
    for i in range(mat.shape[0]):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        if start == end:
            continue
        row_cols = mat.indices[start:end]
        row_data = mat.data[start:end]
        if row_data.size > k:
            keep = np.argpartition(row_data, -k)[-k:]
            keep = keep[np.argsort(-row_data[keep])]
            row_cols = row_cols[keep]
            row_data = row_data[keep]
        rows.extend([i] * len(row_cols))
        cols.extend(row_cols.tolist())
        data.extend(row_data.tolist())
    return sp.csr_matrix((data, (rows, cols)), shape=mat.shape, dtype=np.float32)


def build_edge_transition(src, dst, times, num_nodes: int, edge_neighbor_k: int, beta: float) -> sp.csr_matrix:
    """Backward-compatible alias for the exact temporal edge-event Gamma_T."""
    return build_temporal_edge_event_transition(src, dst, times, num_nodes, edge_neighbor_k, beta)


def build_sf_etrl_expanded_graph(src, dst, num_nodes: int) -> sp.csr_matrix:
    """Build the SF-ETRL subdivision graph: node <-> edge-node <-> node."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    n = int(num_nodes)
    m = int(len(src))
    edge_nodes = n + np.arange(m, dtype=np.int64)
    rows = np.concatenate([src, edge_nodes, dst, edge_nodes])
    cols = np.concatenate([edge_nodes, src, edge_nodes, dst])
    data = np.ones(rows.shape[0], dtype=np.float32)
    graph = sp.csr_matrix((data, (rows, cols)), shape=(n + m, n + m), dtype=np.float32)
    graph.sum_duplicates()
    return graph


def compute_truncated_edge_ppr(P_E: sp.csr_matrix, alpha: float, T: int, edge_ppr_topk: int) -> sp.csr_matrix:
    m = P_E.shape[0]
    Pi = alpha * sp.eye(m, format="csr", dtype=np.float32)
    P_power = sp.eye(m, format="csr", dtype=np.float32)
    for ell in range(1, int(T) + 1):
        P_power = sparse_row_topk(P_power @ P_E, edge_ppr_topk)
        Pi = Pi + (alpha * ((1.0 - alpha) ** ell)) * P_power
        Pi = sparse_row_topk(Pi, edge_ppr_topk)
    return row_normalize(Pi)


def _sample_next(rng: np.random.RandomState, indices: np.ndarray, probs: np.ndarray) -> int:
    if len(indices) == 1:
        return int(indices[0])
    return int(indices[rng.choice(len(indices), p=probs)])


def _row_weighted_choice(P: sp.csr_matrix, row: int, rng: np.random.RandomState) -> int:
    start, end = P.indptr[row], P.indptr[row + 1]
    if start == end:
        return int(row)
    nbrs = P.indices[start:end]
    probs = P.data[start:end].astype(np.float64, copy=False)
    total = float(probs.sum())
    if total <= 0.0:
        return int(row)
    return _sample_next(rng, nbrs, probs / total)


def _build_temporal_incidence(src, dst, times, num_nodes: int):
    incident = [[] for _ in range(num_nodes)]
    for eid, (u, v) in enumerate(zip(src, dst)):
        incident[int(u)].append(int(eid))
        if int(v) != int(u):
            incident[int(v)].append(int(eid))
    positions = [{} for _ in range(num_nodes)]
    for node, events in enumerate(incident):
        events.sort(key=lambda eid: (float(times[eid]), int(eid)))
        positions[node] = {int(eid): pos for pos, eid in enumerate(events)}
    return incident, positions


def _sample_temporal_incident_edge(
    node: int,
    prev_eid: int,
    incident,
    positions,
    times,
    edge_neighbor_k: int,
    beta: float,
    rng: np.random.RandomState,
) -> int:
    events = incident[node]
    if not events:
        return -1
    if prev_eid < 0 or prev_eid not in positions[node]:
        return int(events[rng.randint(len(events))])

    pos = positions[node][prev_eid]
    t0 = float(times[prev_eid])
    candidates = []
    weights = []
    jpos = pos + 1
    limit = int(edge_neighbor_k)
    while jpos < len(events) and (limit <= 0 or len(candidates) < limit):
        eid = int(events[jpos])
        dt_raw = float(times[eid]) - t0
        if dt_raw > 0.0 or (abs(dt_raw) <= EPS_TIME and eid > int(prev_eid)):
            candidates.append(eid)
            weights.append(float(np.exp(-float(beta) * max(dt_raw, 0.0))))
        jpos += 1

    if not candidates:
        return int(prev_eid)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return int(candidates[rng.randint(len(candidates))])
    return int(candidates[rng.choice(len(candidates), p=weights / total)])


def build_temporal_edge_event_transition(src, dst, times, num_nodes: int, edge_neighbor_k: int, beta: float) -> sp.csr_matrix:
    """Build Gamma_T over edge-event states by endpoint-wise temporal successors."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    times = np.asarray(times, dtype=np.float32)
    m = int(len(src))
    if m == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)

    incident, positions = _build_temporal_incidence(src, dst, times, int(num_nodes))
    weights: Dict[Tuple[int, int], float] = defaultdict(float)
    limit = int(edge_neighbor_k)
    beta = float(beta)

    for i in range(m):
        t0 = float(times[i])
        for side in (0, 1):
            node = int(src[i]) if side == 0 else int(dst[i])
            events = incident[node]
            pos = positions[node].get(int(i))
            candidates = []
            raw_weights = []
            if pos is not None:
                jpos = pos + 1
                while jpos < len(events) and (limit <= 0 or len(candidates) < limit):
                    j = int(events[jpos])
                    dt_raw = float(times[j]) - t0
                    if dt_raw > 0.0 or (abs(dt_raw) <= EPS_TIME and j > int(i)):
                        candidates.append(j)
                        raw_weights.append(float(np.exp(-beta * max(dt_raw, 0.0))))
                    jpos += 1

            if not candidates:
                weights[(int(i), int(i))] += 0.5
                continue

            raw_weights = np.asarray(raw_weights, dtype=np.float64)
            total = float(raw_weights.sum())
            if total <= 0.0 or not np.isfinite(total):
                prob = 1.0 / float(len(candidates))
                for j in candidates:
                    weights[(int(i), int(j))] += 0.5 * prob
            else:
                probs = raw_weights / total
                for j, prob in zip(candidates, probs):
                    weights[(int(i), int(j))] += 0.5 * float(prob)

    rows, cols, data = zip(*((i, j, w) for (i, j), w in weights.items()))
    P = sp.csr_matrix((data, (rows, cols)), shape=(m, m), dtype=np.float32)
    P.sum_duplicates()
    rowsum = np.asarray(P.sum(axis=1)).ravel()
    bad = np.where((rowsum <= 0.0) | ~np.isfinite(rowsum))[0]
    if bad.size:
        P = P.tolil()
        P[bad, bad] = 1.0
        P = P.tocsr()
    return row_normalize(P)


def _state_is_edge(state: int, m: int) -> bool:
    return 0 <= int(state) < int(m)


def _decode_intermediate_state(state: int, m: int, src, dst) -> Tuple[int, int, int]:
    tmp = int(state) - int(m)
    prev_eid = tmp // 2
    side = tmp % 2
    node = int(src[prev_eid]) if side == 0 else int(dst[prev_eid])
    return int(prev_eid), int(side), int(node)


def _sample_state_expanded_next(
    state: int,
    src,
    dst,
    times,
    incident,
    positions,
    edge_neighbor_k: int,
    beta: float,
    rng: np.random.RandomState,
) -> int:
    m = int(len(src))
    state = int(state)
    if _state_is_edge(state, m):
        eid = state
        if rng.rand() < 0.5:
            return int(m + 2 * eid)
        return int(m + 2 * eid + 1)

    prev_eid, _side, node = _decode_intermediate_state(state, m, src, dst)
    next_eid = _sample_temporal_incident_edge(
        node, prev_eid, incident, positions, times, edge_neighbor_k, beta, rng
    )
    if next_eid < 0:
        next_eid = prev_eid
    return int(next_eid)


def _init_state_forest_worker(src, dst, times, num_nodes: int, edge_neighbor_k: int, beta: float):
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    times = np.asarray(times, dtype=np.float32)
    incident, positions = _build_temporal_incidence(src, dst, times, int(num_nodes))
    _STATE_FOREST_WORKER.clear()
    _STATE_FOREST_WORKER.update(
        {
            "src": src,
            "dst": dst,
            "times": times,
            "incident": incident,
            "positions": positions,
            "edge_neighbor_k": int(edge_neighbor_k),
            "beta": float(beta),
        }
    )


def _state_expanded_temporal_forest_chunk(args):
    sample_start, sample_count, total_samples, alpha, seed = args
    src = _STATE_FOREST_WORKER["src"]
    dst = _STATE_FOREST_WORKER["dst"]
    times = _STATE_FOREST_WORKER["times"]
    incident = _STATE_FOREST_WORKER["incident"]
    positions = _STATE_FOREST_WORKER["positions"]
    edge_neighbor_k = _STATE_FOREST_WORKER["edge_neighbor_k"]
    beta = _STATE_FOREST_WORKER["beta"]

    m = int(len(src))
    total_states = 3 * m
    alpha_star = 1.0 - np.sqrt(1.0 - float(alpha))
    hit_weight = float(alpha) / alpha_star / float(total_samples)
    in_forest = np.zeros(total_states, dtype=bool)
    root = np.full(total_states, -1, dtype=np.int64)
    rows, cols, data = [], [], []

    for sample_idx in range(sample_start, sample_start + sample_count):
        rng = np.random.RandomState(int(seed) + int(sample_idx) * 1000003)
        in_forest.fill(False)
        root.fill(-1)

        for s in range(total_states):
            if in_forest[s]:
                continue

            u = int(s)
            path = []
            path_pos = {}
            terminal_root = -1

            while True:
                if in_forest[u]:
                    terminal_root = int(root[u])
                    break

                if u in path_pos:
                    keep_end = path_pos[u] + 1
                    for removed in path[keep_end:]:
                        path_pos.pop(int(removed), None)
                    path = path[:keep_end]
                else:
                    path_pos[u] = len(path)
                    path.append(u)

                if rng.rand() < alpha_star:
                    terminal_root = int(u)
                    break

                u = _sample_state_expanded_next(
                    u, src, dst, times, incident, positions, edge_neighbor_k, beta, rng
                )

            for state in path:
                state = int(state)
                in_forest[state] = True
                root[state] = int(terminal_root)

        for i in range(m):
            r = int(root[i])
            if _state_is_edge(r, m):
                rows.append(i)
                cols.append(r)
                data.append(hit_weight)

    if data:
        pi_chunk = sp.csr_matrix((data, (rows, cols)), shape=(m, m), dtype=np.float32)
        pi_chunk.sum_duplicates()
    else:
        pi_chunk = sp.csr_matrix((m, m), dtype=np.float32)
    return sample_start, sample_count, int(len(data)), pi_chunk


def compute_state_expanded_temporal_forest_edge_ppr(
    src,
    dst,
    times,
    num_nodes: int,
    alpha: float,
    forest_samples: int,
    edge_ppr_topk: int,
    edge_neighbor_k: int,
    beta: float,
    seed: int,
) -> sp.csr_matrix:
    """SF-ETRL-inspired temporal state-expanded generalization.

    This implements the state-expanded temporal subdivision forest:
    epsilon_i -> h_{i,x} -> epsilon_j.
    The previous-edge-dependent transition is made first-order by encoding h_{i,x}=(x,e_i).
    The micro-step stopping probability is alpha_star=1-sqrt(1-alpha), so that two micro-steps correspond to one edge-level PPR step:
    (1-alpha_star)^2 = 1-alpha.
    The forest sampling graph has directed weights w_tilde(s,r)=lambda P_T(s,r), lambda=(1-alpha_star)/alpha_star.
    Adding an auxiliary root with weight 1 gives stopping probability alpha_star.
    The estimator counts only edge-event roots and rescales by alpha/alpha_star.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    times = np.asarray(times, dtype=np.float32)
    m = int(len(src))
    if m == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)

    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1) for forest PPR, got {alpha}")
    sample_num = max(1, int(forest_samples))
    alpha_star = 1.0 - np.sqrt(1.0 - alpha)
    hit_weight = alpha / alpha_star / float(sample_num)
    total_states = 3 * m

    workers = max(1, int(os.environ.get("TEMPORAL_FOREST_WORKERS", "1")))
    chunk_size = max(1, int(os.environ.get("TEMPORAL_FOREST_CHUNK_SAMPLES", "1")))
    combine_chunks = max(1, int(os.environ.get("TEMPORAL_FOREST_COMBINE_CHUNKS", "8")))
    if workers > 1 and sample_num > 1:
        tasks = [
            (start, min(chunk_size, sample_num - start), sample_num, alpha, seed)
            for start in range(0, sample_num, chunk_size)
        ]
        progress_start = time.time()
        print(
            f"[temporal_state_forest] start parallel sampling samples={sample_num} "
            f"edge_events={m} total_states={total_states} workers={workers} chunk_samples={chunk_size}",
            flush=True,
        )
        Pi = None
        pending = []
        samples_done = 0
        hits_done = 0
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()
        with ctx.Pool(
            processes=workers,
            initializer=_init_state_forest_worker,
            initargs=(src, dst, times, int(num_nodes), int(edge_neighbor_k), float(beta)),
        ) as pool:
            for _sample_start, sample_count, hit_count, pi_chunk in pool.imap_unordered(
                _state_expanded_temporal_forest_chunk, tasks
            ):
                pending.append(pi_chunk)
                if len(pending) >= combine_chunks:
                    batch = pending[0]
                    for extra in pending[1:]:
                        batch = (batch + extra).tocsr()
                    batch.sum_duplicates()
                    Pi = batch if Pi is None else (Pi + batch).tocsr()
                    Pi.sum_duplicates()
                    pending.clear()
                samples_done += int(sample_count)
                hits_done += int(hit_count)
                elapsed = time.time() - progress_start
                avg = elapsed / max(1, samples_done)
                eta = avg * (sample_num - samples_done)
                print(
                    f"[temporal_state_forest] samples={samples_done}/{sample_num} "
                    f"elapsed={elapsed:.1f}s avg_per_sample={avg:.2f}s eta={eta:.1f}s "
                    f"hits={hits_done} nnz={Pi.nnz if Pi is not None else 0}",
                    flush=True,
                )
        if pending:
            batch = pending[0]
            for extra in pending[1:]:
                batch = (batch + extra).tocsr()
            batch.sum_duplicates()
            Pi = batch if Pi is None else (Pi + batch).tocsr()
            Pi.sum_duplicates()
        if Pi is None:
            Pi = sp.csr_matrix((m, m), dtype=np.float32)
        missing = np.where(np.asarray(Pi.sum(axis=1)).ravel() <= 0)[0]
        if missing.size:
            Pi = Pi.tolil()
            Pi[missing, missing] = 1.0
            Pi = Pi.tocsr()
        Pi = sparse_row_topk(Pi, edge_ppr_topk)
        return Pi.tocsr()

    incident, positions = _build_temporal_incidence(src, dst, times, int(num_nodes))
    rng = np.random.RandomState(int(seed))
    in_forest = np.zeros(total_states, dtype=bool)
    root = np.full(total_states, -1, dtype=np.int64)
    rows, cols, data = [], [], []

    progress_interval = max(1, sample_num // 20)
    progress_start = time.time()
    print(
        f"[temporal_state_forest] start sampling samples={sample_num} "
        f"edge_events={m} total_states={total_states}",
        flush=True,
    )

    for sample_idx in range(sample_num):
        in_forest.fill(False)
        root.fill(-1)

        for s in range(total_states):
            if in_forest[s]:
                continue

            u = int(s)
            path = []
            path_pos = {}
            terminal_root = -1

            while True:
                if in_forest[u]:
                    terminal_root = int(root[u])
                    break

                if u in path_pos:
                    keep_end = path_pos[u] + 1
                    for removed in path[keep_end:]:
                        path_pos.pop(int(removed), None)
                    path = path[:keep_end]
                else:
                    path_pos[u] = len(path)
                    path.append(u)

                if rng.rand() < alpha_star:
                    terminal_root = int(u)
                    break

                u = _sample_state_expanded_next(
                    u, src, dst, times, incident, positions, edge_neighbor_k, beta, rng
                )

            for state in path:
                state = int(state)
                in_forest[state] = True
                root[state] = int(terminal_root)

        for i in range(m):
            r = int(root[i])
            if _state_is_edge(r, m):
                rows.append(i)
                cols.append(r)
                data.append(hit_weight)

        if (sample_idx + 1) == 1 or (sample_idx + 1) % progress_interval == 0 or (sample_idx + 1) == sample_num:
            elapsed = time.time() - progress_start
            done = sample_idx + 1
            avg = elapsed / done
            eta = avg * (sample_num - done)
            print(
                f"[temporal_state_forest] sample={done}/{sample_num} "
                f"elapsed={elapsed:.1f}s avg_per_sample={avg:.2f}s eta={eta:.1f}s hits={len(data)}",
                flush=True,
            )

    if data:
        Pi = sp.csr_matrix((data, (rows, cols)), shape=(m, m), dtype=np.float32)
        Pi.sum_duplicates()
    else:
        Pi = sp.csr_matrix((m, m), dtype=np.float32)
    missing = np.where(np.asarray(Pi.sum(axis=1)).ravel() <= 0)[0]
    if missing.size:
        Pi = Pi.tolil()
        Pi[missing, missing] = 1.0
        Pi = Pi.tocsr()
    Pi = sparse_row_topk(Pi, edge_ppr_topk)
    return Pi.tocsr()


def compute_sf_etrl_temporal_forest_edge_ppr(
    src,
    dst,
    times,
    num_nodes: int,
    alpha: float,
    forest_samples: int,
    edge_ppr_topk: int,
    edge_neighbor_k: int,
    beta: float,
    seed: int,
) -> sp.csr_matrix:
    """Legacy implementation using original node states; not theory-aligned for previous-edge-dependent temporal transitions.

    State space matches SF-ETRL: original nodes 0..n-1 plus event edge nodes
    n..n+m-1.  Walks start from edge nodes, forest roots are sampled on this
    expanded graph, and only edge-node to edge-node root hits are emitted as
    the final m x m proximity matrix.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    times = np.asarray(times, dtype=np.float32)
    n = int(num_nodes)
    m = int(len(src))
    if m == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)

    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1) for forest PPR, got {alpha}")
    sample_num = max(1, int(forest_samples))
    alpha2 = 1.0 - np.sqrt(1.0 - alpha)
    hit_weight = alpha / alpha2 / float(sample_num)

    incident, positions = _build_temporal_incidence(src, dst, times, n)
    rng = np.random.RandomState(int(seed))
    total_nodes = n + m
    in_forests = np.zeros(total_nodes, dtype=bool)
    next_node = np.full(total_nodes, -1, dtype=np.int64)
    root = np.full(total_nodes, -1, dtype=np.int64)
    rows, cols, data = [], [], []

    for _ in range(sample_num):
        in_forests.fill(False)
        next_node.fill(-1)
        root.fill(-1)

        for s in range(n, total_nodes):
            u = int(s)
            prev_eid = int(s - n)
            while not in_forests[u]:
                if rng.rand() < alpha2:
                    in_forests[u] = True
                    root[u] = u
                    if u >= n:
                        rows.append(u - n)
                        cols.append(u - n)
                        data.append(hit_weight)
                    break

                if u >= n:
                    eid = int(u - n)
                    if int(src[eid]) == int(dst[eid]):
                        nxt = int(src[eid])
                    elif rng.rand() < 0.5:
                        nxt = int(src[eid])
                    else:
                        nxt = int(dst[eid])
                    prev_eid = eid
                else:
                    eid = _sample_temporal_incident_edge(
                        int(u), prev_eid, incident, positions, times, edge_neighbor_k, beta, rng
                    )
                    if eid < 0:
                        nxt = int(u)
                    else:
                        nxt = n + int(eid)
                        prev_eid = int(eid)
                next_node[u] = int(nxt)
                u = int(nxt)

            r = int(root[u])
            u = int(s)
            while not in_forests[u]:
                in_forests[u] = True
                root[u] = r
                if u >= n and r >= n:
                    rows.append(u - n)
                    cols.append(r - n)
                    data.append(hit_weight)
                u = int(next_node[u])

    if data:
        Pi = sp.csr_matrix((data, (rows, cols)), shape=(m, m), dtype=np.float32)
        Pi.sum_duplicates()
    else:
        Pi = sp.eye(m, format="csr", dtype=np.float32)
    missing = np.where(np.asarray(Pi.sum(axis=1)).ravel() <= 0)[0]
    if missing.size:
        Pi = Pi.tolil()
        Pi[missing, missing] = 1.0
        Pi = Pi.tocsr()
    return Pi.tocsr()


def compute_forest_edge_ppr(
    P_E: sp.csr_matrix,
    alpha: float,
    forest_samples: int,
    edge_ppr_topk: int,
    seed: int,
) -> sp.csr_matrix:
    """Legacy forest sampler on an already materialized edge-edge transition."""
    P_E = P_E.tocsr()
    m = P_E.shape[0]
    if m == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)

    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1) for forest PPR, got {alpha}")
    sample_num = max(1, int(forest_samples))
    alpha2 = 1.0 - np.sqrt(1.0 - alpha)
    hit_weight = alpha / alpha2 / float(sample_num)

    rng = np.random.RandomState(int(seed))
    rows, cols, data = [], [], []
    in_forests = np.zeros(m, dtype=bool)
    next_node = np.full(m, -1, dtype=np.int64)
    root = np.full(m, -1, dtype=np.int64)

    for _ in range(sample_num):
        in_forests.fill(False)
        next_node.fill(-1)
        root.fill(-1)

        for s in range(m):
            u = int(s)
            while not in_forests[u]:
                if rng.rand() < alpha2:
                    in_forests[u] = True
                    root[u] = u
                    rows.append(u)
                    cols.append(u)
                    data.append(hit_weight)
                    break
                next_node[u] = _row_weighted_choice(P_E, u, rng)
                u = int(next_node[u])

            r = int(root[u])
            u = int(s)
            while not in_forests[u]:
                in_forests[u] = True
                root[u] = r
                rows.append(u)
                cols.append(r)
                data.append(hit_weight)
                u = int(next_node[u])

    if data:
        Pi = sp.csr_matrix((data, (rows, cols)), shape=(m, m), dtype=np.float32)
        Pi.sum_duplicates()
    else:
        Pi = sp.eye(m, format="csr", dtype=np.float32)
    missing = np.where(np.asarray(Pi.sum(axis=1)).ravel() <= 0)[0]
    if missing.size:
        Pi = Pi.tolil()
        Pi[missing, missing] = 1.0
        Pi = Pi.tocsr()
    Pi = sparse_row_topk(Pi, edge_ppr_topk)
    return Pi.tocsr()


def build_edge_ncut_affinity(Pi_E: sp.csr_matrix, edge_ppr_topk: int) -> sp.csr_matrix:
    W = 0.5 * (Pi_E.tocsr() + Pi_E.T.tocsr())
    W = W.tocsr()
    W.sum_duplicates()
    W.setdiag(0.0)
    W.eliminate_zeros()
    W = sparse_row_topk(W, edge_ppr_topk)
    W.sum_duplicates()
    return W.tocsr()


def compute_edge_ppr_cached(
    dataset: str,
    src,
    dst,
    times,
    num_nodes: int,
    cache_dir: str,
    method: str,
    alpha: float,
    T: int,
    forest_samples: int,
    edge_neighbor_k: int,
    edge_ppr_topk: int,
    beta: float,
    seed: int,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, dict]:
    method = str(method)
    forest_impl = {
        "temporal_state_forest": "state_expanded_temporal_subdivision_forest",
        "forest": "state_expanded_temporal_subdivision_forest_alias",
        "legacy_temporal_forest": "legacy_original_node_temporal_forest",
        "truncated": "temporal_edge_event_truncated_ppr",
    }.get(method)
    if forest_impl is None:
        raise ValueError(f"Unsupported edge_ppr_method: {method}")

    cfg = {
        "dataset": dataset,
        "method": method,
        "alpha": float(alpha),
        "T": int(T),
        "forest_samples": int(forest_samples),
        "edge_neighbor_k": int(edge_neighbor_k),
        "edge_ppr_topk": int(edge_ppr_topk),
        "beta": float(beta),
        "seed": int(seed),
        "num_events": int(len(src)),
        "forest_impl": forest_impl,
    }
    cfg_hash = hash_cfg(cfg)
    ds_cache = os.path.join(cache_dir, dataset)
    ensure_dir(ds_cache)
    p_path = os.path.join(ds_cache, f"edge_transition_{cfg_hash}.npz")
    pi_path = os.path.join(ds_cache, f"edge_ppr_{method}_{cfg_hash}.npz")
    w_path = os.path.join(ds_cache, f"edge_ncut_affinity_{method}_{cfg_hash}.npz")
    meta_path = os.path.join(ds_cache, f"edge_ppr_{method}_{cfg_hash}.json")

    if method in ["temporal_state_forest", "forest", "truncated"]:
        if os.path.exists(p_path):
            P_E = sp.load_npz(p_path).tocsr()
        else:
            P_E = build_temporal_edge_event_transition(src, dst, times, num_nodes, edge_neighbor_k, beta)
            sp.save_npz(p_path, P_E)
    elif method == "legacy_temporal_forest":
        if os.path.exists(p_path):
            P_E = sp.load_npz(p_path).tocsr()
        else:
            P_E = build_sf_etrl_expanded_graph(src, dst, num_nodes)
            sp.save_npz(p_path, P_E)

    if os.path.exists(pi_path) and os.path.exists(w_path):
        Pi_E = sp.load_npz(pi_path).tocsr()
        W_E = sp.load_npz(w_path).tocsr()
    else:
        if method in ["temporal_state_forest", "forest"]:
            Pi_E = compute_state_expanded_temporal_forest_edge_ppr(
                src=src,
                dst=dst,
                times=times,
                num_nodes=num_nodes,
                alpha=alpha,
                forest_samples=forest_samples,
                edge_ppr_topk=edge_ppr_topk,
                edge_neighbor_k=edge_neighbor_k,
                beta=beta,
                seed=seed,
            )
        elif method == "legacy_temporal_forest":
            Pi_E = compute_sf_etrl_temporal_forest_edge_ppr(
                src=src,
                dst=dst,
                times=times,
                num_nodes=num_nodes,
                alpha=alpha,
                forest_samples=forest_samples,
                edge_ppr_topk=edge_ppr_topk,
                edge_neighbor_k=edge_neighbor_k,
                beta=beta,
                seed=seed,
            )
        elif method == "truncated":
            Pi_E = compute_truncated_edge_ppr(P_E, alpha, T, edge_ppr_topk)
        W_E = build_edge_ncut_affinity(Pi_E, edge_ppr_topk)
        sp.save_npz(pi_path, Pi_E)
        sp.save_npz(w_path, W_E)
        with open(meta_path, "w", encoding="utf-8") as writer:
            json.dump(cfg, writer, indent=2, sort_keys=True)

    stats = {
        "config_hash": cfg_hash,
        "P_shape": P_E.shape,
        "P_nnz": int(P_E.nnz),
        "P_avg_outdegree": float(P_E.nnz / max(1, P_E.shape[0])),
        "Pi_shape": Pi_E.shape,
        "Pi_nnz": int(Pi_E.nnz),
        "Pi_avg_row_nnz": float(Pi_E.nnz / max(1, Pi_E.shape[0])),
        "W_shape": W_E.shape,
        "W_nnz": int(W_E.nnz),
    }
    return P_E, Pi_E, W_E, stats
