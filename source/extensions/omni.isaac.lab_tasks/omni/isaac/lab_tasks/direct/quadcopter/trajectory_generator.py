# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Polynomial / Legendre reference trajectory sampling for QuadcopterTrajectoryEnv."""

from __future__ import annotations

import numpy as np
import torch
from numpy import polynomial as P
from numpy.polynomial.legendre import Legendre

# Max Legendre degree in the task basis (inclusive). JSON rows are padded/truncated to this width.
TRAJ_LEGENDRE_MAX_DEGREE = 6
TRAJ_POLY_NUM_COEFFS = TRAJ_LEGENDRE_MAX_DEGREE + 1


def _pad_power_basis(poly_coef: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros(width, dtype=np.float64)
    n = min(int(poly_coef.size), width)
    if n:
        out[:n] = np.asarray(poly_coef, dtype=np.float64).reshape(-1)[:n]
    return out


def _pad_legendre_basis(poly: Legendre, width: int) -> np.ndarray:
    c = np.asarray(poly.coef, dtype=np.float64).reshape(-1)
    out = np.zeros(width, dtype=np.float64)
    n = min(c.size, width)
    if n:
        out[:n] = c[:n]
    return out


def _coeff_row_one_hot(index: int) -> list[int]:
    r = [0] * TRAJ_POLY_NUM_COEFFS
    r[index] = 1
    return r


class PolynomialTrajectoryGenerator:
    def __init__(
        self,
        device,
        num_envs,
        max_traj_dur=10,
        freq=100,
        vn=0.5,
        mode=0,
        num_trials=1000,
        eval_mode=False,
        predef_coeff=None,
    ):
        self.N = int(max_traj_dur * freq)
        self.vn = vn
        self.H = max_traj_dur
        self.device = device

        self.B = self.vn * self.H

        self.mode = mode
        self.curr_experiment = 0
        self.curr_experiment_tracker = torch.zeros(num_envs).long().to(self.device)

        self.eval = eval_mode

        z = [0] * TRAJ_POLY_NUM_COEFFS

        if mode == 0:
            self.coefficients = torch.tensor([z])
            self.legendre = torch.zeros_like(self.coefficients)

        elif mode == 1:
            self.coefficients = torch.tensor([_coeff_row_one_hot(1)])
            self.legendre = torch.zeros_like(self.coefficients)

        elif mode == 2:
            self.coefficients = torch.tensor([_coeff_row_one_hot(2)])
            self.legendre = torch.zeros_like(self.coefficients)

        elif mode == 3:
            self.coefficients = torch.tensor(
                [
                    z,
                    _coeff_row_one_hot(1),
                    _coeff_row_one_hot(2),
                ]
            )
            self.legendre = torch.zeros_like(self.coefficients)

        elif mode == 4:
            self.coefficients = torch.tensor([z])
            self.legendre = torch.tensor([z])

        elif mode == 5:
            self.coefficients = []
            self.legendre = []
            for deg in range(6):
                coeff, poly = self.generate_legendre_coeffecients(deg, returnpoly=True)
                self.coefficients.append(coeff)
                self.legendre.append(_pad_legendre_basis(poly, TRAJ_POLY_NUM_COEFFS))

            self.coefficients = torch.from_numpy(np.stack(self.coefficients, axis=0)).float()
            self.legendre = torch.from_numpy(np.stack(self.legendre, axis=0)).float()

        elif mode == 6:
            coeff, poly = self.generate_legendre_coeffecients(6, returnpoly=True)
            leg = _pad_legendre_basis(poly, TRAJ_POLY_NUM_COEFFS)
            self.coefficients = torch.from_numpy(np.asarray(coeff, dtype=np.float32).reshape(1, -1)).float()
            self.legendre = torch.from_numpy(np.asarray(leg, dtype=np.float32).reshape(1, -1)).float()

        elif mode == 7:
            coeff_rows: list[torch.Tensor] = []
            leg_rows: list[torch.Tensor] = []
            if predef_coeff is not None:
                for coeff in predef_coeff:
                    coeffs, poly = self.convert_predefined_coeffecients(coeff, returnpoly=True)
                    leg = _pad_legendre_basis(poly, TRAJ_POLY_NUM_COEFFS)
                    coeff_rows.append(torch.tensor(coeffs, dtype=torch.float32))
                    leg_rows.append(torch.tensor(leg, dtype=torch.float32))
            else:
                for _ in range(num_trials):
                    coeffs, poly = self.generate_legendre_coeffecients(5, eval=True, returnpoly=True)
                    leg = _pad_legendre_basis(poly, TRAJ_POLY_NUM_COEFFS)
                    coeff_rows.append(torch.tensor(coeffs, dtype=torch.float32))
                    leg_rows.append(torch.tensor(leg, dtype=torch.float32))
            if not coeff_rows:
                raise ValueError("mode 7 requires non-empty predef_coeff or num_trials > 0")
            self.coefficients = torch.stack(coeff_rows, dim=0)
            self.legendre = torch.stack(leg_rows, dim=0)

        self.coefficients = self.coefficients.to(self.device)
        self.legendre = self.legendre.to(self.device)

    def activate_eval_mode(self):
        self.eval = True

    def generate_legendre_coeffecients(self, deg, eval=False, returnpoly=False):
        if deg < 0 or deg > TRAJ_LEGENDRE_MAX_DEGREE:
            raise ValueError(f"deg must be in [0, {TRAJ_LEGENDRE_MAX_DEGREE}], got {deg}")
        coeffs = np.zeros(TRAJ_POLY_NUM_COEFFS)
        coeffs[deg] = 1

        if eval:
            coeffs[: deg + 1] = np.random.randn(deg + 1)

        legendre_poly = Legendre(coeffs, domain=[0, self.B])
        poly_coef = legendre_poly.convert(kind=P.Polynomial).coef
        coefficients = _pad_power_basis(poly_coef, TRAJ_POLY_NUM_COEFFS)
        coefficients[0] = 0

        if returnpoly:
            return coefficients, legendre_poly
        return coefficients

    def convert_predefined_coeffecients(self, coeffs, returnpoly=False):
        c = np.asarray(coeffs, dtype=np.float64).ravel()
        if c.size > TRAJ_POLY_NUM_COEFFS:
            raise ValueError(
                f"Predefined Legendre row length {c.size} exceeds {TRAJ_POLY_NUM_COEFFS} "
                f"(max degree {TRAJ_LEGENDRE_MAX_DEGREE})."
            )
        c = np.pad(c, (0, TRAJ_POLY_NUM_COEFFS - c.size))
        legendre_poly = Legendre(c, domain=[0, self.B])
        poly_coef = legendre_poly.convert(kind=P.Polynomial).coef
        coefficients = _pad_power_basis(poly_coef, TRAJ_POLY_NUM_COEFFS)
        coefficients[0] = 0

        if returnpoly:
            return coefficients, legendre_poly
        return coefficients

    def generate_tasks_random(
        self,
        lvl,
        num_environments,
        env_ids,
        forced_task_indices: torch.Tensor | None = None,
    ):
        n_coeff = self.coefficients.shape[0]
        cap = max(1, min(int(lvl), int(n_coeff))) if n_coeff > 0 else 1
        if forced_task_indices is not None:
            random_indices = forced_task_indices.long().to(self.device).reshape(num_environments)
            self.curr_experiment_tracker[env_ids] = random_indices
        else:
            random_indices = torch.randint(0, cap, (num_environments,), device=self.device)
            if self.eval:
                random_indices = (
                    torch.arange(
                        self.curr_experiment,
                        self.curr_experiment + num_environments,
                        device=self.device,
                    )
                    % n_coeff
                )
                self.curr_experiment = (self.curr_experiment + num_environments) % max(n_coeff, 1)
                self.curr_experiment_tracker[env_ids] = random_indices.to(self.device)

        selected_tasks = self.legendre[random_indices].float()
        selected_coeffs = self.coefficients[random_indices]
        return selected_coeffs, selected_tasks

    def naive_random_sample_tasks(self, num_environments):
        tasks = np.random.randn(num_environments, TRAJ_POLY_NUM_COEFFS)
        tasks[:, -1] = 0
        legendre_polys = [Legendre(task, domain=[0, self.B]) for task in tasks]
        coeffs = [
            _pad_power_basis(poly.convert(kind=P.Polynomial).coef, TRAJ_POLY_NUM_COEFFS) for poly in legendre_polys
        ]

        selected_coeffs = torch.tensor(coeffs, device=self.device).float()
        selected_coeffs[:, 0] = 0
        selected_tasks = torch.tensor(
            [_pad_legendre_basis(poly, TRAJ_POLY_NUM_COEFFS) for poly in legendre_polys],
            device=self.device,
        ).float()
        selected_tasks[:, 0] = 0
        return selected_coeffs, selected_tasks

    def generate_trajectory(
        self,
        rpose,
        num_environments,
        env_ids,
        offset_r=0.05,
        lvl=10,
        forced_task_indices: torch.Tensor | None = None,
    ):
        pos0 = torch.rand(num_environments, 1, 3, device=self.device) * offset_r + rpose.unsqueeze(1)
        traj_ = torch.zeros(num_environments, self.N, 3, device=self.device)
        traj_[:, :, 0] = torch.linspace(0, self.H * self.vn, self.N, device=self.device)

        selected_coeffs, selected_tasks = self.generate_tasks_random(
            lvl, num_environments, env_ids, forced_task_indices=forced_task_indices
        )

        x = traj_[:, :, 0]

        powers = torch.arange(selected_coeffs.shape[1], device=self.device)
        x_powers = x.unsqueeze(2).pow(powers)
        traj_[:, :, 1] = torch.sum(selected_coeffs.unsqueeze(1) * x_powers, dim=2)

        deriv_powers = powers[:-1]
        deriv_coeffs = selected_coeffs[:, 1:] * powers[1:]

        vx = torch.ones_like(x) * self.vn
        vy = torch.sum(deriv_coeffs.unsqueeze(1) * x.unsqueeze(2).pow(deriv_powers), dim=2)
        vz = torch.zeros_like(vx)

        velocities = torch.stack([vx, vy, vz], dim=2)

        traj = torch.repeat_interleave(pos0, self.N, dim=1) + traj_

        return traj, velocities, selected_tasks
