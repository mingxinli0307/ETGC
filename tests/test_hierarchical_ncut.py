"""Dataset-free checks for the hierarchical ETGC experiment."""
import ast
import csv
import inspect
import json
import os
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge_losses import (cform_ncut_loss, hierarchical_ncut_terms,
                         trace_mincut_orthogonality_loss, projection_loss_global,
                         project_edge_assignments_to_nodes_global,
                         edge_expected_structural_score_gain_loss_global,
                         edge_ppr_proximity_loss_preindexed)
from edge_model import EdgeHiNoSModel, power_sharpen
from edge_main import build_parser, main
from edge_train import EdgeHiNoSTrainer


def matrices(dtype=torch.float64):
    gen = torch.Generator().manual_seed(123)
    Pi = torch.rand(20, 20, generator=gen, dtype=dtype)
    Pi = (Pi + Pi.T) / 2
    Pi.fill_diagonal_(0)
    z1 = torch.randn(20, 6, generator=gen, dtype=dtype, requires_grad=True)
    z2 = torch.randn(6, 3, generator=gen, dtype=dtype, requires_grad=True)
    q1 = z1.softmax(1)
    p1 = power_sharpen(q1)
    q2 = z2.softmax(1)
    return Pi, z1, z2, q1, p1, q2


def test_shapes_stochastic_and_exact_power():
    Pi, z1, z2, q1, p1, q2 = matrices()
    final = p1 @ q2
    for tensor, shape in ((q1, (20, 6)), (p1, (20, 6)), (q2, (6, 3)), (final, (20, 3))):
        assert tensor.shape == shape
        torch.testing.assert_close(tensor.sum(1), torch.ones(shape[0], dtype=tensor.dtype))
    torch.testing.assert_close(p1, q1.square() / q1.square().sum(1, keepdim=True))
    torch.testing.assert_close(power_sharpen(q1, 1.0), q1)


@pytest.mark.parametrize('kind', ['dense', 'coo', 'csr', 'scipy'])
def test_fine_coarse_formula_and_diagonal(kind):
    Pi, _, _, q1, p1, q2 = matrices()
    d = Pi.sum(1)
    affinity = {'dense': Pi, 'coo': Pi.to_sparse(), 'csr': Pi.to_sparse_csr(),
                'scipy': sp.csr_matrix(Pi.numpy())}[kind]
    fine, coarse, Pi_H, degree_H, diagnostics = hierarchical_ncut_terms(q1, p1, q2, affinity, d, row_block_size=7)
    expected = p1.T @ Pi @ p1
    torch.testing.assert_close(Pi_H, expected)
    torch.testing.assert_close(degree_H, expected.sum(1))
    assert torch.all(Pi_H.diagonal() > 0)
    for Q, W, degree, actual in ((p1, Pi, d, fine), (q2, expected, expected.sum(1), coarse)):
        A = Q.T @ torch.diag(degree) @ Q + 1e-8 * torch.eye(Q.shape[1], dtype=Q.dtype)
        B = Q.T @ (torch.diag(degree) - W) @ Q
        torch.testing.assert_close(actual, torch.linalg.solve(A, B).trace())
    assert all(torch.isfinite(x) for x in (fine, coarse,
        trace_mincut_orthogonality_loss(p1, 6), trace_mincut_orthogonality_loss(q2, 3)))


def test_hard_partition_equivalence():
    Pi, _, _, _, _, q2 = matrices()
    hard = torch.nn.functional.one_hot(torch.arange(20) % 6, 6).double()
    d = Pi.sum(1)
    Pi_H = hard.T @ Pi @ hard
    D_graph = torch.diag(Pi_H.sum(1))
    D_projected = hard.T @ torch.diag(d) @ hard
    torch.testing.assert_close(D_graph, D_projected, atol=1e-12, rtol=1e-12)
    direct, _ = cform_ncut_loss(hard @ q2, Pi, d)
    coarse, _ = cform_ncut_loss(q2, Pi_H, Pi_H.sum(1))
    error = abs(float((direct - coarse).detach()))
    assert error < 1e-10
    print(f'hard_equivalence_absolute_error={error:.12g}')


def test_soft_discrepancy_psd_trace_and_sharpening():
    Pi, _, _, q1, p1, q2 = matrices()
    d = Pi.sum(1)
    _, _, Pi_H, degree_H, stats = hierarchical_ncut_terms(q1, p1, q2, Pi, d)
    delta = torch.diag(degree_H) - p1.T @ torch.diag(d) @ p1
    min_eig = float(torch.linalg.eigvalsh((delta + delta.T) / 2).min().detach())
    trace_target = (d * (1 - p1.square().sum(1))).sum()
    error = abs(float((delta.trace() - trace_target).detach()))
    assert min_eig >= -1e-6
    torch.testing.assert_close(delta.trace(), trace_target, atol=1e-10, rtol=1e-10)
    assert stats['p1_softness'] <= stats['q1_softness'] + 1e-12
    print(f'soft_delta_min_eigenvalue={min_eig:.12g} trace_error={error:.12g} stats={stats}')


def test_coarse_gradient_chain_and_finite_differences():
    Pi, z1, z2, q1, p1, q2 = matrices()
    _, coarse, Pi_H, _, _ = hierarchical_ncut_terms(q1, p1, q2, Pi, Pi.sum(1))
    grads = torch.autograd.grad(coarse, (Pi_H, p1, q1, z1, z2), retain_graph=True)
    for grad in grads:
        assert torch.isfinite(grad).all() and grad.norm() > 0
    print('coarse_gradient_norms=' + str([float(g.norm()) for g in grads]))
    def loss(first, second):
        soft = first.softmax(1)
        sharp = power_sharpen(soft)
        return hierarchical_ncut_terms(soft, sharp, second.softmax(1), Pi, Pi.sum(1))[1]
    assert torch.autograd.gradcheck(loss, (z1, z2), fast_mode=True, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('head', ['legacy_mlp', 'cosine_prototype'])
def test_model_global_backward_and_routing(head):
    torch.manual_seed(81)
    model = EdgeHiNoSModel(np.random.RandomState(1).normal(size=(8, 5)).astype('float32'),
                          time_dim=2, edge_dim=8, edge_hidden_dim=10, cluster_hidden_dim=9,
                          K=3, directed=False, hier_ncut_h=6, cluster_head_type=head)
    src = torch.arange(20) % 8
    dst = (src + 1) % 8
    t = torch.randn(20, 2)
    out = model.forward_hierarchical(src, dst, t)
    assert out['z1_all'].shape == (20, 6)
    assert model.cluster_output.weight.shape[0] == 6
    assert model.coarse_assignment_logits.shape == (6, 3)
    torch.testing.assert_close(model(src, dst, t)[1], out['q_final_all'])
    Pi = matrices()[0].float()
    trainer = object.__new__(EdgeHiNoSTrainer)
    trainer.args = SimpleNamespace(lambda_fine_ncut=0.7, lambda_coarse_ncut=1.3,
        lambda_edge_ncut=0.6, lambda_orth=0.2, lambda_esg=0.4, lambda_proj=0.3,
        global_ncut_row_block_size=7)
    trainer.H, trainer.K = 6, 3
    trainer.Pi_cut_sparse_torch = Pi.to_sparse()
    trainer.D_Pi_degree_np = Pi.sum(1).numpy()
    trainer.src_t, trainer.dst_t = src, dst
    trainer.data = SimpleNamespace(num_nodes=8)
    with mock.patch('edge_train.edge_expected_structural_score_gain_loss_global',
                    wraps=edge_expected_structural_score_gain_loss_global) as esg, \
         mock.patch('edge_train.projection_loss_global', wraps=projection_loss_global) as projection:
        loss, terms, stats = trainer.compute_global_objective(out)
        assert esg.call_args.args[0] is out['p1_all']
        assert projection.call_args.args[0] is out['q_final_all']
    expected = (0.6 * (0.7 * terms['fine_ncut_loss'] + 1.3 * terms['coarse_ncut_loss'])
                + 0.2 * (terms['fine_orth_loss'] + terms['coarse_orth_loss'])
                + 0.4 * terms['esg_loss'] + 0.3 * terms['projection_loss'])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for param in (model.node_emb, model.cluster_output.weight, model.coarse_assignment_logits):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm() > 0
    print(f'{head}_global_loss={float(loss.detach()):.10g} head_grad={float(model.cluster_output.weight.grad.norm()):.10g} '
          f'coarse_logits_grad={float(model.coarse_assignment_logits.grad.norm()):.10g}')


@pytest.mark.parametrize('encoder', ['direct_node_time', 'mlp'])
def test_proximity_step_updates_hierarchical_shared_representation_only(encoder):
    torch.manual_seed(91)
    model = EdgeHiNoSModel(
        np.random.RandomState(4).normal(size=(8, 5)).astype('float32'),
        time_dim=2, edge_dim=7, edge_hidden_dim=9, cluster_hidden_dim=6,
        K=3, directed=False, hier_ncut_h=6, edge_encoder_mode=encoder,
        cluster_head_type='cosine_prototype',
    )
    src = torch.tensor([0, 1, 2, 3, 4, 5])
    dst = torch.tensor([1, 2, 3, 4, 5, 6])
    time_feat = torch.randn(6, 2)
    before_repr = model.forward_hierarchical(src, dst, time_feat)['edge_repr_all'].detach().clone()
    before_first_head = model.cluster_output.weight.detach().clone()
    before_second_level = model.coarse_assignment_logits.detach().clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    r_all = model.encode_edge_events(src, dst, time_feat)
    proximity_loss = edge_ppr_proximity_loss_preindexed(
        r_all,
        anchors=torch.tensor([0, 1, 2]),
        positives=torch.tensor([1, 2, 3]),
        negatives=torch.tensor([5, 4, 5]),
        weights=torch.ones(3),
        similarity_mode='event_dot',
    )
    assert torch.isfinite(proximity_loss)
    optimizer.zero_grad(set_to_none=True)
    proximity_loss.backward()
    assert model.node_emb.grad is not None and model.node_emb.grad.norm() > 0
    if encoder == 'mlp':
        edge_mlp_grads = [p.grad for p in model.edge_mlp.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in edge_mlp_grads)
        assert sum(float(g.norm()) for g in edge_mlp_grads) > 0
    assert model.cluster_output.weight.grad is None
    assert model.coarse_assignment_logits.grad is None
    optimizer.step()

    after_repr = model.forward_hierarchical(src, dst, time_feat)['edge_repr_all'].detach()
    assert not torch.allclose(before_repr, after_repr)
    torch.testing.assert_close(model.cluster_output.weight, before_first_head)
    torch.testing.assert_close(model.coarse_assignment_logits, before_second_level)


def test_cli_validation_and_default_dimensions():
    args = build_parser().parse_args([])
    assert (args.hier_ncut_h, args.sharpen_gamma, args.lambda_fine_ncut, args.lambda_coarse_ncut) == (-1, 2., 1., 1.)
    kwargs = dict(initial_node_features=np.ones((8, 4)), time_dim=2, edge_dim=6,
                  edge_hidden_dim=8, cluster_hidden_dim=5, K=3, directed=False)
    assert EdgeHiNoSModel(**kwargs).H == 6
    for h in (1, 2):
        with pytest.raises(ValueError):
            EdgeHiNoSModel(**kwargs, hier_ncut_h=h)
    for gamma in (0.9, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            EdgeHiNoSModel(**kwargs, sharpen_gamma=gamma)
    names = vars(args)
    removed = ('node_prior', 'node_sbm', 'anchor', 'lambda_bal', 'orth_type',
               'global_cut_scale', 'global_orth_scale', 'prototype_init',
               'prototype_seed', 'cluster_init_mode', 'initialization_state')
    assert not any(any(token in name for token in removed) for name in names)


def test_repeated_events_project_independently():
    q = torch.tensor([[0.9, .1], [.2, .8], [.3, .7]])
    src, dst = torch.tensor([0, 0, 1]), torch.tensor([1, 1, 2])
    S = project_edge_assignments_to_nodes_global(q, src, dst, 3)
    expected = torch.stack(((q[0] + q[1]) / 2, q.mean(0), q[2]))
    torch.testing.assert_close(S, expected)


@pytest.mark.parametrize('head,warmup', [('legacy_mlp', 0),
    ('cosine_prototype', 0), ('cosine_prototype', 1)])
def test_dataset_free_training_smoke(tmp_path, head, warmup):
    data = tmp_path / 'dataset' / 'toy'
    data.mkdir(parents=True)
    # Includes repeated endpoints: all six events must survive loading.
    (data / 'toy.txt').write_text('0 1 0\n0 1 1\n1 2 2\n2 3 3\n3 0 4\n1 3 5\n')
    (data / 'node2label.txt').write_text('0 0\n1 0\n2 1\n3 1\n')
    output = tmp_path / 'run'
    args = build_parser().parse_args(['--dataset', 'toy', '--device', 'cpu', '--data_root', str(data.parent),
        '--cache_dir', str(tmp_path / 'cache'), '--output_dir', str(output), '--epoch', str(warmup + 1),
        '--batch_size', '2', '--edge_ppr_method', 'truncated', '--edge_ppr_topk', '-1',
        '--edge_dim', '8', '--time_dim', '3', '--edge_hidden_dim', '10', '--cluster_hidden_dim', '6',
        '--cluster_head_type', head, '--node_emb_mode', 'full',
        '--lambda_proj', '0.1', '--prox_warmup_epochs', str(warmup), '--global_q_chunk_size', '2', '--quiet', '1'])
    main(args)
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'success' and (result['M'], result['H'], result['K']) == (6, 4, 2)
    assert set(('ACC', 'Macro_F1', 'NMI', 'ARI')) <= result['final_metrics'].keys()
    with (output / 'metrics.csv').open() as reader:
        records = list(csv.DictReader(reader))
    assert int(records[-1]['prox_optimizer_steps']) > 0
    for name in ('fine_ncut_loss', 'coarse_ncut_loss', 'orth_loss', 'esg_loss', 'projection_loss', 'global_loss'):
        assert np.isfinite(float(records[-1][name]))
    if warmup:
        assert records[0]['global_loss'] == ''
    assert (output / 'diagnostic.json').is_file()


def test_near_uniform_and_isolated_affinity_are_finite():
    Pi, _, _, q1, _, _ = matrices(torch.float32)
    for W in (Pi, torch.zeros_like(Pi)):
        first = torch.full((20, 6), 1 / 6, requires_grad=True)
        second = torch.zeros(6, 3, requires_grad=True)
        fine, coarse, _, _, _ = hierarchical_ncut_terms(first, power_sharpen(first), second.softmax(1), W, W.sum(1))
        (fine + coarse).backward()
        assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()
