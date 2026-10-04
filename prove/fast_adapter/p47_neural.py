"""P47: fixed, clipped Gauss--Newton, and direct same-latent adaptation.

The direct arm minimizes the *same* ray quadratic at the current neural offset.
It does not fit independent corners or replace the learned decoder. For each
corner k, d_sk = (reference_k - C_sk) / scale + offset_k + B_k z. Thus
H = mean_s sum_k B_k.T A_sk B_k and g = mean_s sum_k B_k.T A_sk d_sk.
The update is z' = z - H^+ g: the minimum-change solution in the retained
eigenspace. The declared numerical rank threshold reflects float32 public ray
matrices, so this is a numerical minimum, not an exact-arithmetic claim.

Unlike continuous's three damped, clipped steps, direct has no damping or step
bound. A large direct jump is an algorithmic difference and is recorded. Neural
encoding/decoding and persisted z remain float32; only the small solve is CPU
float64. No model parameter, observation, correspondence, or support selection
is optimized by the direct solver.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from prove.fast_adapter.geometry3d_adapter import initial
from prove.fast_adapter.neural_geometry_adapter import (
    GEOMETRY_SCALE, align_observation, ray_matrices,
)
from prove.fast_adapter.p45_neural import (
    P45Latent, TASK_GEOMETRY, features, load_model,
)

ARMS = ('fixed', 'continuous', 'direct')
REPRESENTATIONS = ('geometry', 'relative_fusion')
SOLVER_CONTRACT = {
    'schema': 'prove.p47-same-latent-solver.v1',
    'objective': 'P45 ray_loss at current encoded offset and frozen learned B',
    'method': 'symmetric_eigendecomposition_minimum_change_pseudoinverse',
    'compute_dtype': 'float64',
    'compute_device': 'cpu',
    'runtime_dtype': 'float32',
    'rank_rtol': 1e-6,
    'rank_atol': 0.0,
    'retained_eigenvalues': 'lambda > rank_rtol * max(abs(lambda))',
    'negative_eigenvalue_policy': 'raise if lambda_min < -rank_threshold',
    'nullspace_policy': 'preserve incoming z in discarded eigenspace',
    'stationarity_tolerance': 'rank_rtol * (norm(g_before) + norm(H,2)*norm(delta_z))',
    'stationarity_policy': 'record residual and tolerance; no outcome-dependent fallback',
    'damping': 0.0,
    'step_clip': None,
    'direct_solves_per_update': 1,
    'minimum_support_views': 3,
    'support': {'cap': 32, 'prefix': 12, 'recent': 20},
    'failure_policy': 'raise on nonfinite inputs/outputs or materially indefinite quadratic',
}


def direct_solver_contract():
    """Return an independent JSON-safe copy for an experiment freeze binding."""
    return copy.deepcopy(SOLVER_CONTRACT)


class DirectSolveError(RuntimeError):
    """A failed solve, carrying JSON-safe numerical diagnostics for the runner."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def direct_quadratic_step(model, reference, offset, z, A, C):
    """Return ``(z_next, diagnostics)`` for one runtime episode (batch size 1).

    Coefficients are taken from the existing float32 tensors, with no new rays
    or regularizer. Symmetrization removes only the antisymmetric part, which
    cannot contribute to a real quadratic. The CPU solve does not use autograd;
    the returned detached tensor uses the input z's dtype/device. Diagnostics
    distinguish the float64 solution from its persisted runtime rounding.
    """
    started = time.perf_counter()

    def array(value):
        return value.detach().to(device='cpu', dtype=torch.float64).numpy()

    ref, off, prior, a, c, basis = map(array, (
        reference, offset, z, A, C, model.latent_basis,
    ))
    dim = model.latent_dim
    if (ref.shape != (1, 4, 3) or off.shape != (1, 12)
            or prior.shape != (1, dim) or basis.shape != (12, dim)
            or a.ndim != 5 or a.shape[0] != 1 or a.shape[2:] != (4, 3, 3)
            or a.shape[1] < 1 or c.shape != a.shape[:-1]):
        raise ValueError('Direct solve requires one episode and P45 ray/decoder shapes')
    diagnostics = dict(method=SOLVER_CONTRACT['method'], compute_dtype='float64',
                       rank_rtol=SOLVER_CONTRACT['rank_rtol'])
    if not all(np.isfinite(value).all() for value in (ref, off, prior, a, c, basis)):
        raise DirectSolveError('Direct solve received nonfinite coefficients', diagnostics)
    # Keep A itself, rather than A.T @ A: stored float32 rays need not be exact
    # projectors, and squaring A would change the legacy objective.
    a = (a[0] + a[0].swapaxes(-1, -2)) * .5
    b = basis.reshape(4, 3, dim)
    origin = (ref[0][None] - c[0]) / GEOMETRY_SCALE + off[0].reshape(1, 4, 3)
    residual = origin + np.einsum('kid,d->ki', b, prior[0])[None]
    hessian = np.einsum('kid,skij,kje->de', b, a, b) / len(a)
    hessian = (hessian + hessian.T) * .5
    gradient = np.einsum('kid,skij,skj->d', b, a, residual) / len(a)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    scale = float(np.max(np.abs(eigenvalues)))
    threshold = SOLVER_CONTRACT['rank_rtol'] * scale
    diagnostics.update(hessian_eigenvalues=eigenvalues.tolist(), rank_threshold=threshold)
    if eigenvalues[0] < -threshold:
        diagnostics['solve_seconds'] = time.perf_counter() - started
        raise DirectSolveError('Direct ray quadratic is materially indefinite', diagnostics)
    retained = eigenvalues > threshold
    rank = int(retained.sum())
    coordinates = eigenvectors.T @ gradient
    change = -(eigenvectors[:, retained] @ (coordinates[retained] / eigenvalues[retained]))
    candidate = prior[0] + change
    returned = torch.as_tensor(candidate[None], dtype=z.dtype, device=z.device).detach()
    stored = array(returned)[0]
    if not np.isfinite(candidate).all() or not np.isfinite(stored).all():
        raise DirectSolveError('Direct solution is not finite in runtime precision', diagnostics)

    def loss(value):
        difference = origin + np.einsum('kid,d->ki', b, value)[None]
        return float(.5 * np.einsum('ski,skij,skj->', difference, a, difference) / len(a))

    gradient_norm = float(np.linalg.norm(gradient))
    delta_norm = float(np.linalg.norm(change))
    stationarity = float(np.linalg.norm(hessian @ change + gradient))
    tolerance = SOLVER_CONTRACT['rank_rtol'] * (gradient_norm + scale * delta_norm)
    retained_condition = float(eigenvalues[-1] / eigenvalues[retained][0]) if rank else None
    diagnostics.update(
        rank=rank, latent_dim=dim, rank_deficient=rank < dim,
        condition_number=retained_condition if rank == dim else None,
        retained_condition_number=retained_condition,
        loss_before=loss(prior[0]), loss_after_float64=loss(candidate),
        loss_after_runtime_z=loss(stored),
        gradient_norm=gradient_norm, delta_z_norm=delta_norm,
        stored_delta_z_norm=float(np.linalg.norm(stored - prior[0])),
        z_norm=float(np.linalg.norm(stored)),
        nullspace_gradient_norm=float(np.linalg.norm(coordinates[~retained])),
        stationarity_residual=stationarity, stationarity_tolerance=tolerance,
        stationarity_within_tolerance=stationarity <= tolerance,
        runtime_rounding_norm=float(np.linalg.norm(stored - candidate)),
        solve_seconds=time.perf_counter() - started,
    )
    return returned, diagnostics


class P47Latent(P45Latent):
    """P45 runtime interface, with unchanged fixed/continuous implementations."""

    def __init__(self, model, arm='continuous', device='cpu'):
        if arm not in ARMS:
            raise ValueError(f'Unknown P47 update arm: {arm}')
        if model.representation not in REPRESENTATIONS:
            raise ValueError('P47 only compares geometry and relative_fusion')
        super().__init__(model, 'fixed' if arm == 'direct' else arm, device)
        self.arm = arm

    def update(self, obs, t):
        if self.arm != 'direct':
            return super().update(obs, t)
        if obs['corners'] is None:
            return
        if self.reference is None:
            self.reference = initial(obs, self.model.geometry_task)
            aligned = align_observation(obs, self.reference)
            self.first = features(aligned, self.model.representation)
        aligned = align_observation(obs, self.points if self.points is not None else self.reference)
        current = features(aligned, self.model.representation)
        reference = self.tensor(self.reference)[None]
        with torch.no_grad():
            z0, offset = self.model.encode(self.tensor(self.first)[None], self.tensor(current)[None], reference)
        if self.z is None:
            self.z = z0.detach()
            self.initial_z = self.z.clone()
        before = self.z.clone()
        fixed_points = self.model.decode(reference, offset, self.z).detach()
        self.history.append(ray_matrices(aligned))
        support = self.history if len(self.history) <= 32 else self.history[:12] + self.history[-20:]
        A = self.tensor(np.stack([entry[0] for entry in support]))[None]
        C = self.tensor(np.stack([entry[1] for entry in support]))[None]
        loss_before = float(self.model.ray_loss(fixed_points, A, C)[0])
        inner, solver = [], None
        if len(self.history) >= SOLVER_CONTRACT['minimum_support_views']:
            self.z, solver = direct_quadratic_step(self.model, reference, offset, self.z, A, C)
            inner = [dict(loss=solver['loss_before'], gradient=solver['gradient_norm'],
                          delta=solver['stored_delta_z_norm'], method='direct')]
            self.update_events += 1
        self.points = self.model.decode(reference, offset, self.z).detach().cpu().numpy()[0]
        loss_after = float(self.model.ray_loss(self.tensor(self.points)[None], A, C)[0])
        self.records.append(dict(
            t=int(t), S=len(support), U=len(inner), update_events=self.update_events,
            z=self.z.cpu().numpy()[0].tolist(), initial_z=self.initial_z.cpu().numpy()[0].tolist(),
            delta_z=(self.z - self.initial_z).cpu().numpy()[0].tolist(), donor_t=None,
            latent_change=float((self.z - before).norm()), z_from_initial=float((self.z - self.initial_z).norm()),
            initial_z_points=self.model.decode(reference, offset, self.initial_z).detach().cpu().numpy()[0].tolist(),
            point_change=float(np.linalg.norm(self.points - fixed_points.cpu().numpy()[0])),
            support_loss_before=loss_before, support_loss_after=loss_after, inner=inner,
            pre_update_points=fixed_points.cpu().numpy()[0].tolist(), representation=self.model.representation,
            solver=solver,
        ))
