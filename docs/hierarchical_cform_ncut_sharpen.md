# ETGC hierarchical sharpened C-form Ncut experiment

Base: `exp/etgc-mainline-refine`, commit `8ce33be`.
Experiment: `exp/hierarchical-cform-ncut-sharpen`.
The original server worktree and its untracked sweep script are preserved.

## Implemented

Temporal interactions remain separate edge events. The data loader, temporal
PPR generation, original affinity diagonal handling, symmetric affinity
construction, ESG formula, proximity sampling, and proximity loss are unchanged.
In the equations below, `Pi_E` means the existing symmetric `Pi_cut` used for
the cut objective; the raw directed temporal PPR remains the proximity input.

`K` is the number of unique labels in `node2label.txt`. With `hier_ncut_h <= 0`,
`H = 2K`; otherwise use the requested `H`. Reject `H < K`. Reject nonfinite
`sharpen_gamma` or values below 1. The default gamma is 2.

For either the legacy MLP head or cosine prototype head:

```text
Z1 = head(edge_repr)                          [M,H]
Q1 = softmax(Z1, dim=1)                       [M,H]
P1 = Q1^gamma / sum(Q1^gamma, dim=1)          [M,H]
Pi_H = P1.T @ Pi_E @ P1                      [H,H]
d_H = Pi_H.sum(dim=1)                        [H]
Q2 = softmax(coarse_assignment_logits, dim=1) [H,K]
Q_final = P1 @ Q2                            [M,K]
S_final = RowNorm(incidence_NxM @ Q_final)    [N,K]
pred_y = argmax(S_final, dim=1)
```

The code uses a clamped denominator for power sharpening. There is no frequency
correction or detached assignment. Coarse diagonal entries are retained, and no
coarse sparsification or normalization is performed. Final assignments are not
passed through another softmax. `coarse_assignment_logits` is initialized once
with `normal_(mean=0, std=0.02)`. Existing optional first-head initialization
uses `H`; no new clustering initializer is introduced.

Define the pure C-form:

```text
CForm(Q, Pi, d) = trace(solve(Q.T @ (d[:,None]*Q) + eps*I,
                             Q.T @ (d[:,None]*Q - Pi@Q)))
fine_ncut_loss = CForm(P1, Pi_E, Pi_E.sum(1))
coarse_ncut_loss = CForm(Q2, Pi_H, Pi_H.sum(1))
Orth(Q,C) = ||Q.T Q / ||Q.T Q||_F - I_C/sqrt(C)||_F
```

Sparse/dense affinity products and the C-form solve accumulate in float64 so
`eps=1e-8` remains effective for near-uniform assignments. These casts preserve
autograd. The fine affinity product is reused when constructing `Pi_H`.

The actual global objective in `edge_train.py` is:

```python
hierarchical_cut_loss = (
    args.lambda_fine_ncut * fine_ncut_loss
    + args.lambda_coarse_ncut * coarse_ncut_loss
)
orth_loss = (
    fine_orth_loss
    + coarse_orth_loss
)
global_loss = (
    args.lambda_edge_ncut * hierarchical_cut_loss
    + args.lambda_orth * orth_loss
    + args.lambda_esg * esg_loss
    + args.lambda_proj * projection_loss
)
```

Here `fine_orth_loss = Orth(P1,H)`, `coarse_orth_loss = Orth(Q2,K)`, ESG uses
`P1`, and projection uses `Q_final`. Proximity still takes separate batch
optimizer steps and retains its warmup behavior; it is absent from global loss.

Diagnostics record degree-weighted Q1/P1 softness and
`trace(diag(Pi_H.sum(1)) - P1.T @ D_E @ P1)`. They do not enter the objective.
Warmup epochs leave uncomputed global fields blank.

## Removed / deprecated

Removed the node embedding anchor, node SBM, node prior loss and logit injection,
direct prior evaluation, balance, OrthQA, independent cut/orth glue scales, and
legacy single-layer cut training paths. Their CLI, log fields, diagnostic paths,
and dependent experiment launchers/tests were removed rather than disabled with
zero coefficients. The old large diagnostic module and unused prior module were
also removed. Historical implementations and experiment commands remain on the
base branch and in Git history. Historical documents describe that base, not
this branch's supported commands.

The standalone node-embedding KMeans baseline remains separate; its shared
inventory utilities were extracted into `scripts/node_embedding_inventory.py`.
It is not used by hierarchical ETGC. Output embeddings now use the requested
fresh run directory (`etgc_Q_final.npy`, `etgc_S_final.npy`).

## Experimentally verified scope

Only mathematical/unit checks and a one-epoch smoke run have been performed.
There is no full training or multi-dataset performance claim.

- Hard-partition C-form equivalence absolute error: `4.4408920985e-16`.
- Soft discrepancy minimum eigenvalue: `-4.28971365807e-16`.
- Soft discrepancy trace identity absolute error: `0`.
- Degree-weighted softness: Q1 `0.6868876762`, P1 `0.4995658070`.
- Coarse gradient norms along Pi_H, P1, Q1, Z1, Z2:
  `0.05744782, 0.08045691, 0.09790120, 0.01970943, 0.00854302`.
- Finite-difference gradcheck and both cluster-head global backward checks pass.
- Dense, sparse COO, sparse CSR, and scipy block affinity checks pass.
- Existing proximity pair/RNG/loss/gradient regression checks pass.
- Snapshot roundtrip restores both levels and validates H/gamma metadata.

School smoke uses all `188508` events, `H=18`, `K=9`, default forest50 affinity
copied from the existing cache, one epoch, zero proximity warmup, and
`lambda_proj=0.1` to exercise projection. It completed 369 proximity optimizer
steps and one global optimizer step on `cuda:2`; all losses were finite.
Its evidence directory is `/tmp/etgc_hier_validation_1ex2_e1l` on the server,
including `smoke.log`, `school_smoke/result.json`, `metrics.csv`, `diagnostic.json`,
`config.json`, and validation logs. This temporary evidence is also downloaded
to the local `results/server_runs` directory during handoff.

## Reproduce focused validation

```bash
python -m py_compile edge_model.py edge_losses.py edge_train.py edge_main.py tests/test_hierarchical_ncut.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests -q
```

For a future smoke run, explicitly set `--epoch 1 --prox_warmup_epochs 0`, valid
data/embedding/cache paths, and a new `--output_dir`. The default warmup is still
five epochs, so a default one-epoch command would test only proximity.
