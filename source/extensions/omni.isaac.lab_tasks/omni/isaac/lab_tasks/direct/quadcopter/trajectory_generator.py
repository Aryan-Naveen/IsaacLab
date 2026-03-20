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

        if mode == 0:
            self.coefficients = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
            self.legendre = torch.zeros_like(self.coefficients)

        elif mode == 1:
            self.coefficients = torch.tensor([[0, 1, 0, 0, 0, 0, 0]])
            self.legendre = torch.zeros_like(self.coefficients)

        elif mode == 2:
            self.coefficients = torch.tensor([[0, 0, 1, 0, 0, 0, 0]])
            self.legendre = torch.zeros_like(self.coefficients)

        elif mode == 3:
            self.coefficients = torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                ]
            )
            self.legendre = torch.zeros_like(self.coefficients)

        elif mode == 4:
            self.coefficients = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
            self.legendre = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])

        elif mode == 5:
            self.coefficients = []
            self.legendre = []
            for deg in range(6):
                coeff, poly = self.generate_legendre_coeffecients(deg, returnpoly=True)
                self.coefficients.append(coeff)
                self.legendre.append(poly.coef)

            self.coefficients = torch.tensor(self.coefficients).float()
            self.legendre = torch.tensor(self.legendre).float()

        elif mode == 6:
            coeff, poly = self.generate_legendre_coeffecients(6, returnpoly=True)
            self.coefficients = torch.tensor([coeff]).float()
            self.legendre = torch.tensor([poly.coef]).float()

        elif mode == 7:
            self.coefficients = torch.tensor([])
            self.legendre = torch.tensor([])
            if predef_coeff is not None:
                for coeff in predef_coeff:
                    coeffs, poly = self.convert_predefined_coeffecients(coeff, returnpoly=True)
                    self.coefficients = torch.cat([self.coefficients, torch.tensor(coeffs).unsqueeze(0)], dim=0)
                    self.legendre = torch.cat([self.legendre, torch.tensor(poly.coef).unsqueeze(0)], dim=0)
            else:
                for _ in range(num_trials):
                    coeffs, poly = self.generate_legendre_coeffecients(5, eval=True, returnpoly=True)
                    self.coefficients = torch.cat([self.coefficients, torch.tensor(coeffs).unsqueeze(0)], dim=0)
                    self.legendre = torch.cat([self.legendre, torch.tensor(poly.coef).unsqueeze(0)], dim=0)

            self.coefficients = self.coefficients.float()
            self.legendre = self.legendre.float()

        self.coefficients = self.coefficients.to(self.device)
        self.legendre = self.legendre.to(self.device)

    def activate_eval_mode(self):
        self.eval = True

    def generate_legendre_coeffecients(self, deg, eval=False, returnpoly=False):
        coeffs = np.zeros(7)
        coeffs[deg] = 1

        if eval:
            coeffs[: deg + 1] = np.random.randn(deg + 1)

        legendre_poly = Legendre(coeffs, domain=[0, self.B])
        coeffs = np.pad(legendre_poly.convert(kind=P.Polynomial).coef, (0, 6 - deg))
        coeffs[0] = 0

        if returnpoly:
            return coeffs, legendre_poly
        return coeffs

    def convert_predefined_coeffecients(self, coeffs, returnpoly=False):
        legendre_poly = Legendre(coeffs, domain=[0, self.B])
        coeffs = legendre_poly.convert(kind=P.Polynomial).coef

        coefficients = np.zeros(6)
        coefficients[1 : coeffs.size] = coeffs[1:]

        if returnpoly:
            return coefficients, legendre_poly
        return coefficients

    def generate_tasks_random(self, lvl, num_environments, env_ids):
        random_indices = torch.randint(0, min(lvl, self.coefficients.shape[0]), (num_environments,), device=self.device)
        if self.eval:
            random_indices = (
                torch.arange(
                    self.curr_experiment,
                    self.curr_experiment + num_environments,
                    device=self.device,
                )
                % self.coefficients.shape[0]
            )
            self.curr_experiment = (self.curr_experiment + num_environments) % self.coefficients.shape[0]
            self.curr_experiment_tracker[env_ids] = random_indices.to(self.device)

        selected_tasks = self.legendre[random_indices].float()
        selected_coeffs = self.coefficients[random_indices]
        return selected_coeffs, selected_tasks

    def naive_random_sample_tasks(self, num_environments):
        tasks = np.random.randn(num_environments, 7)
        tasks[:, -1] = 0
        legendre_polys = [Legendre(task, domain=[0, self.B]) for task in tasks]
        coeffs = [np.pad(poly.convert(kind=P.Polynomial).coef, (0, 7 - poly.convert(kind=P.Polynomial).coef.size)) for poly in legendre_polys]

        selected_coeffs = torch.tensor(coeffs, device=self.device).float()
        selected_coeffs[:, 0] = 0
        selected_tasks = torch.tensor([poly.coef for poly in legendre_polys], device=self.device).float()
        selected_tasks[:, 0] = 0
        return selected_coeffs, selected_tasks

    def generate_trajectory(self, rpose, num_environments, env_ids, offset_r=0.05, lvl=10):
        pos0 = torch.rand(num_environments, 1, 3, device=self.device) * offset_r + rpose.unsqueeze(1)
        traj_ = torch.zeros(num_environments, self.N, 3, device=self.device)
        traj_[:, :, 0] = torch.linspace(0, self.H * self.vn, self.N, device=self.device)

        if lvl == -3:
            selected_coeffs, selected_tasks = self.naive_random_sample_tasks(num_environments)
        else:
            selected_coeffs, selected_tasks = self.generate_tasks_random(lvl, num_environments, env_ids)

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
