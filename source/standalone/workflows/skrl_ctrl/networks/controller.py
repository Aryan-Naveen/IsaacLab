import numpy as np
import math

import torch
from scipy.spatial.transform import Rotation
import torch.nn as nn

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class MLP(nn.Module):
    def __init__(self,
                                input_dim,
                                hidden_dim,
                                output_dim,
                                hidden_depth,
                                output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                                            output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, hidden_activation=nn.ELU(inplace=False), output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), hidden_activation] # inplace=True
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), hidden_activation] # inplace=True
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def pybullet_getEulerFromQuaternion_torch(q):
    """
    q: Tensor of shape (N, 4) in PyBullet format [w, x, y, z]
    returns: Tensor (N, 3) -> [roll, pitch, yaw]
    """
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    w = q[:, 0]

    sqx = x * x
    sqy = y * y
    sqz = z * z
    squ = w * w

    # sarg = -2*(x*z - w*y)
    sarg = -2.0 * (x * z - w * y)

    # Allocate outputs
    roll  = torch.zeros_like(x)
    pitch = torch.zeros_like(x)
    yaw   = torch.zeros_like(x)

    # Gimbal lock masks
    mask_neg = sarg <= -0.99999
    mask_pos = sarg >=  0.99999
    mask_mid = ~(mask_neg | mask_pos)

    # ---- sarg <= -0.99999 ----
    roll[mask_neg]  = 0.0
    pitch[mask_neg] = -0.5 * math.pi
    yaw[mask_neg]   = 2.0 * torch.atan2(x[mask_neg], -y[mask_neg])

    # ---- sarg >= 0.99999 ----
    roll[mask_pos]  = 0.0
    pitch[mask_pos] = 0.5 * math.pi
    yaw[mask_pos]   = 2.0 * torch.atan2(-x[mask_pos], y[mask_pos])

    # ---- normal case ----
    roll[mask_mid] = torch.atan2(
        2.0 * (y[mask_mid] * z[mask_mid] + w[mask_mid] * x[mask_mid]),
        squ[mask_mid] - sqx[mask_mid] - sqy[mask_mid] + sqz[mask_mid]
    )

    pitch[mask_mid] = torch.asin(sarg[mask_mid])

    yaw[mask_mid] = torch.atan2(
        2.0 * (x[mask_mid] * y[mask_mid] + w[mask_mid] * z[mask_mid]),
        squ[mask_mid] + sqx[mask_mid] - sqy[mask_mid] - sqz[mask_mid]
    )

    return torch.stack([roll, pitch, yaw], dim=1)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


class PIDController(nn.Module):
    """PID control class for Crazyflies (Online and Offline compatible)."""

    ################################################################################

    cf_mass = 0.027  # kg
    GRAVITY = 9.81 * cf_mass
    KF = 3.16e-10
    moment_scale = 0.01

    def __init__(self, num_environments, device, integral_horizon=None):
        """
        Args:
            num_environments: Number of envs (batch size for online)
            device: torch device
            integral_horizon: int or None. 
                              If int > 0, uses a fixed history buffer of this length 
                              for computing integrals (Online and Offline).
                              If None, uses standard infinite accumulation.
        """
        super().__init__()

        self.encode_lambda = 0.1
        # -------------------------------
        # Force PID coefficients
        # -------------------------------
        self.P_COEFF_FOR = torch.tensor([0.4, 0.4, 1.25], device=device, dtype=torch.float32)
        self.I_COEFF_FOR = torch.tensor([0.05, 0.05, 0.05], device=device, dtype=torch.float32)
        self.D_COEFF_FOR = torch.tensor([0.2, 0.2, 0.5], device=device, dtype=torch.float32)

        # -------------------------------
        # Torque PID coefficients
        # -------------------------------
        self.P_COEFF_TOR = torch.tensor([70000.0, 70000.0, 60000.0], device=device, dtype=torch.float32)
        self.I_COEFF_TOR = torch.tensor([0.0, 0.0, 500.0], device=device, dtype=torch.float32)
        self.D_COEFF_TOR = torch.tensor([20000.0, 20000.0, 12000.0], device=device, dtype=torch.float32)

        # -------------------------------
        # Motor / PWM constants
        # -------------------------------
        self.PWM2RPM_SCALE = torch.tensor(0.2685, device=device, dtype=torch.float32)
        self.PWM2RPM_CONST = torch.tensor(4070.3, device=device, dtype=torch.float32)
        self.MIN_PWM = torch.tensor(20000.0, device=device, dtype=torch.float32)
        self.MAX_PWM = torch.tensor(65535.0, device=device, dtype=torch.float32)

        # -------------------------------
        # Mixer matrix
        # -------------------------------
        self.MIXER_MATRIX = torch.tensor(
            [[-0.5, -0.5, -1.0],
             [-0.5,  0.5,  1.0],
             [ 0.5,  0.5, -1.0],
             [ 0.5, -0.5,  1.0]],
            device=device, dtype=torch.float32
        )

        self.n_envs = num_environments
        self.gravity = torch.tensor([0.0, 0.0, self.GRAVITY], device=device, dtype=torch.float32)
        self.device = device
        self.integral_horizon = integral_horizon

        # -------------------------------
        # Persistent Internal State (Online only)
        # -------------------------------
        # State tensors will be initialized in reset()
        self.integral_pos_e = None
        self.integral_rpy_e = None
        self.last_rpy = torch.zeros((self.n_envs, 3), dtype=torch.float32, device=self.device)

        # -------------------------------
        # Task Encoder
        # -------------------------------
        self.task_encoder = nn.Sequential(
            nn.Linear(90, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 9)
        ).to(device)

        self.reset()

    def reset(self):
        """Resets the internal control states (Online mode)."""
        self.last_rpy = torch.zeros((self.n_envs, 3), dtype=torch.float32, device=self.device)
        
        if self.integral_horizon is not None and self.integral_horizon > 0:
            # Fixed Horizon: Initialize as buffer (N, H, 3)
            self.integral_pos_e = torch.zeros((self.n_envs, self.integral_horizon, 3), 
                                              dtype=torch.float32, device=self.device)
            self.integral_rpy_e = torch.zeros((self.n_envs, self.integral_horizon, 3), 
                                              dtype=torch.float32, device=self.device)
        else:
            # Standard: Initialize as accumulator (N, 3)
            self.integral_pos_e = torch.zeros((self.n_envs, 3), dtype=torch.float32, device=self.device)
            self.integral_rpy_e = torch.zeros((self.n_envs, 3), dtype=torch.float32, device=self.device)

    def reset_idx(self, idxs):
        """Resets specific environment indices (Online mode)."""
        if not torch.is_tensor(idxs):
            idxs = torch.tensor(idxs, device=self.device, dtype=torch.long)
        
        # This works for both (N, 3) and (N, H, 3) due to broadcasting 0.0
        self.integral_pos_e[idxs] = 0.0
        self.integral_rpy_e[idxs] = 0.0
        self.last_rpy[idxs] = 0.0

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, 
                       target_pos, target_rpy, target_vel, reset,
                       prior_pos_integral=None,
                       prior_rpy_integral=None,
                       prior_last_rpy=None):
        """
        Calculates control actions.
        - If prior_* args are None: Uses internal self.integral_* states (Online).
        - If prior_* args are Tensors: Uses those states without updating internal logic (Offline).
        """
        
        # --- 1. RESET LOGIC ---
        use_internal = prior_pos_integral is None

        if use_internal:
            # Online: Reset internal states directly
            self.reset_from_mask(reset)
            current_pos_state = self.integral_pos_e
            current_rpy_state = self.integral_rpy_e
            current_last_rpy = self.last_rpy
            
            # Since reset_from_mask already zeroed them out, we don't need to mask here.
            reset_mask_float = None
        else:
            # Offline: Functional reset using masking
            reset_mask_float = (reset > 0.5).float().view(-1, 1) # (B, 1)

            # Broadcaster for integral state (handle B,H,3 vs B,3)
            if prior_pos_integral.dim() == 3:
                mask_broad = reset_mask_float.unsqueeze(1) # (B, 1, 1)
            else:
                mask_broad = reset_mask_float # (B, 1)

            # Apply reset mask
            current_pos_state = prior_pos_integral * (1.0 - mask_broad)
            current_rpy_state = prior_rpy_integral * (1.0 - mask_broad)
            
            current_last_rpy = prior_last_rpy 
            if current_last_rpy is not None:
                current_last_rpy = current_last_rpy * (1.0 - reset_mask_float)

        # --- 2. POSITION CONTROL ---
        thrust, computed_target_rpy, pos_e, Ft, next_pos_state = self._dslPIDPositionControl(
            control_timestep, cur_pos, cur_quat, cur_vel, target_pos, target_rpy, target_vel,
            current_integral_state=current_pos_state
        )

        # --- 3. ATTITUDE CONTROL ---
        rpm, target_torques, next_rpy_state, next_last_rpy = self._dslPIDAttitudeControl(
            control_timestep, thrust, cur_quat, computed_target_rpy,
            current_integral_state=current_rpy_state,
            last_rpy_val=current_last_rpy,
            reset_mask=reset_mask_float
        )

        # --- 4. UPDATE INTERNAL STATE (Online Only) ---
        if use_internal:
            # self.integral_pos_e = next_pos_state
            # self.integral_rpy_e = next_rpy_state
            # self.last_rpy = next_last_rpy
            self.integral_pos_e = next_pos_state.detach()
            self.integral_rpy_e = next_rpy_state.detach()
            self.last_rpy = next_last_rpy.detach()            
            # For online, we just return the action
            output = torch.cat([Ft.unsqueeze(1), target_torques], dim=1).squeeze()
            return output
        else:
            # For offline, return action + next states
            output = torch.cat([Ft.unsqueeze(1), target_torques], dim=1).squeeze()
            return output, next_pos_state, next_rpy_state, next_last_rpy

    def reset_from_mask(self, reset):
        if reset.dim() == 2: reset = reset.squeeze(-1)
        idxs = torch.nonzero(reset, as_tuple=False).squeeze(-1)
        if idxs.numel() > 0: self.reset_idx(idxs)

    def _dslPIDPositionControl(self, control_timestep, cur_pos, cur_quat, cur_vel, 
                               target_pos, target_rpy, target_vel, current_integral_state):
        
        N = cur_pos.shape[0]
        
        # Quaternion -> Rotation Matrix
        w, x, y, z = cur_quat.unbind(dim=1)
        xx, yy, zz, ww = x*x, y*y, z*z, w*w
        xy, xz, yz, wx, wy, wz = x*y, x*z, y*z, w*x, w*y, w*z
        cur_rotation = torch.stack([
            ww + xx - yy - zz, 2*(xy - wz),     2*(xz + wy),
            2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx),
            2*(xz - wy),       2*(yz + wx),     ww - xx - yy + zz
        ], dim=1).reshape(N, 3, 3)

        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel

        # -----------------------------------------------------
        # INTEGRAL LOGIC (Fixed Horizon Buffer vs Accumulator)
        # -----------------------------------------------------
        if current_integral_state.dim() == 3:
            # Case: Fixed Horizon Buffer (N, H, 3)
            # Update Buffer: Shift left, append new error at end
            next_state = torch.cat([current_integral_state[:, 1:, :], pos_e.unsqueeze(1)], dim=1)
            
            # Compute Sum for PID
            integral_term = torch.sum(next_state, dim=1) * control_timestep
        else:
            # Case: Standard Accumulator (N, 3)
            integral_term = current_integral_state + pos_e * control_timestep
            next_state = integral_term  # In this mode, state is the sum (before clamp)
        
        # Anti-Windup / Clamping
        # We apply clamping to the term used for PID calculation.
        # Note: If using accumulator mode, we must clamp the state itself for consistency.
        integral_term = torch.clamp(integral_term, -2.0, 2.0)
        
        # Z-axis specific clamping
        int_xy = integral_term[:, :2]
        int_z = torch.clamp(integral_term[:, 2:3], -0.15, 0.15)
        integral_term = torch.cat([int_xy, int_z], dim=1)

        # For accumulator mode, update the state with the clamped value
        if current_integral_state.dim() == 2:
            next_state = integral_term
        # -----------------------------------------------------

        # PID Force
        target_thrust = (
            self.P_COEFF_FOR * pos_e
            + self.I_COEFF_FOR * integral_term
            + self.D_COEFF_FOR * vel_e
        )
        target_thrust[:, 2] += self.GRAVITY

        # Thrust Projection
        body_z = cur_rotation[:, :, 2]
        scalar_thrust = torch.sum(target_thrust * body_z, dim=1)
        scalar_thrust = torch.clamp(scalar_thrust, min=0.0)
        
        Ft = (2 * scalar_thrust) / (1.9 * self.gravity[2]) - 1
        thrust = (torch.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        # Desired Orientation
        target_z_ax = target_thrust / (torch.norm(target_thrust, dim=1, keepdim=True) + 1e-6)
        yaw = target_rpy[:, 2]
        target_x_c = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=1)
        cross_z_xc = torch.cross(target_z_ax, target_x_c, dim=1)
        target_y_ax = cross_z_xc / (torch.norm(cross_z_xc, dim=1, keepdim=True) + 1e-6)
        target_x_ax = torch.cross(target_y_ax, target_z_ax, dim=1)
        target_rotation = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=2)
        
        # Matrix -> Euler
        r = target_rotation
        sy = torch.sqrt(r[:, 0, 0]**2 + r[:, 1, 0]**2)
        singular = sy < 1e-6
        roll = torch.where(singular, torch.atan2(-r[:, 1, 2], r[:, 1, 1]), torch.atan2(r[:, 2, 1], r[:, 2, 2]))
        pitch = torch.where(singular, torch.atan2(-r[:, 2, 0], sy), torch.atan2(-r[:, 2, 0], sy))
        yaw = torch.where(singular, torch.zeros_like(sy), torch.atan2(r[:, 1, 0], r[:, 0, 0]))
        target_euler = torch.stack([roll, pitch, yaw], dim=1)

        return thrust, target_euler, pos_e, Ft, next_state

    def _dslPIDAttitudeControl(self, control_timestep, thrust, cur_quat, target_euler,
                               current_integral_state, last_rpy_val, reset_mask=None):
        
        if thrust.dim() == 2: thrust = thrust.squeeze(-1)
        N = cur_quat.shape[0]

        # Quaternion -> Matrix
        w, x, y, z = cur_quat.unbind(dim=1)
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        cur_rotation = torch.stack([
            1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),
            2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),
            2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)
        ], dim=1).reshape(N, 3, 3)

        # Current RPY
        sy = torch.sqrt(cur_rotation[:, 0, 0]**2 + cur_rotation[:, 1, 0]**2)
        singular = sy < 1e-6
        roll = torch.where(singular, torch.atan2(-cur_rotation[:, 1, 2], cur_rotation[:, 1, 1]), torch.atan2(cur_rotation[:, 2, 1], cur_rotation[:, 2, 2]))
        pitch = torch.where(singular, torch.atan2(-cur_rotation[:, 2, 0], sy), torch.atan2(-cur_rotation[:, 2, 0], sy))
        yaw = torch.where(singular, torch.atan2(cur_rotation[:, 1, 0], cur_rotation[:, 0, 0]), torch.atan2(cur_rotation[:, 1, 0], cur_rotation[:, 0, 0]))
        cur_rpy = torch.stack([roll, pitch, yaw], dim=1)

        # Target Matrix
        cr, cp, cy = torch.cos(target_euler).unbind(dim=1)
        sr, sp, sy_ = torch.sin(target_euler).unbind(dim=1)
        target_rotation = torch.stack([
            cy*cp, cy*sp*sr - sy_*cr, cy*sp*cr + sy_*sr,
            sy_*cp, sy_*sp*sr + cy*cr, sy_*sp*cr - cy*sr,
            -sp,   cp*sr,             cp*cr
        ], dim=1).reshape(N, 3, 3)

        # Rotation Error
        RtR = torch.matmul(target_rotation.transpose(1, 2), cur_rotation)
        RRT = torch.matmul(cur_rotation.transpose(1, 2), target_rotation)
        rot_matrix_e = RtR - RRT
        rot_e = torch.stack([rot_matrix_e[:, 2, 1], rot_matrix_e[:, 0, 2], rot_matrix_e[:, 1, 0]], dim=1)

        # --- RATE ERROR HANDLING ---
        if last_rpy_val is None:
            last_rpy_val = cur_rpy
        elif reset_mask is not None:
            last_rpy_val = cur_rpy * reset_mask + last_rpy_val * (1.0 - reset_mask)
            
        rpy_rates = (cur_rpy - last_rpy_val) / control_timestep
        rpy_rates_e = -rpy_rates
        next_last_rpy = cur_rpy.detach()

        # -----------------------------------------------------
        # INTEGRAL LOGIC
        # -----------------------------------------------------
        # Note: Original code accumulated (-rot_e * dt). 
        # So we store (-rot_e) in buffer or accumulate it.
        integral_input_error = -rot_e

        if current_integral_state.dim() == 3:
            # Case: Fixed Horizon Buffer (N, H, 3)
            # Update Buffer
            next_state = torch.cat([current_integral_state[:, 1:, :], integral_input_error.unsqueeze(1)], dim=1)
            # Compute Sum
            integral_term = torch.sum(next_state, dim=1) * control_timestep
        else:
            # Case: Standard Accumulator (N, 3)
            integral_term = current_integral_state + integral_input_error * control_timestep
            next_state = integral_term

        # Clamp Integral Term
        integral_term = torch.clamp(integral_term, -1500.0, 1500.0)
        int_xy = torch.clamp(integral_term[:, :2], -1.0, 1.0)
        int_z = torch.clamp(integral_term[:, 2:3], -1500.0, 1500.0)
        integral_term = torch.cat([int_xy, int_z], dim=1)
        
        if current_integral_state.dim() == 2:
            next_state = integral_term
        # -----------------------------------------------------

        # PID Torques
        target_torques = (
            -self.P_COEFF_TOR * rot_e
            + self.D_COEFF_TOR * rpy_rates_e
            + self.I_COEFF_TOR * integral_term
        )
        target_torques = torch.clamp(target_torques, -3200.0, 3200.0)

        # Mixer
        pwm = thrust.unsqueeze(1) + torch.matmul(self.MIXER_MATRIX, target_torques.unsqueeze(-1)).squeeze(-1)
        pwm = torch.clamp(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        return rpm, torch.tanh(target_torques/self.moment_scale), next_state, next_last_rpy

    def forward(self, obs):
        """
        Forward handles both:
        1. Online: (N, Obs) -> Uses self.integral buffers
        2. Offline: (B, T, Obs) -> Calculates integrals over T dimension
        """
        # Input Parsing
        B, total_dim = obs.shape
        if total_dim == 101:
            return self._forward_online(obs)
        elif total_dim == 106:
            return self._forward_offline_trajectory(obs)
        else:
            raise ValueError(f"Unsupported observation shape: {obs.shape}")

    def _forward_online(self, obs):
        cur_pos = obs[:, 0:3]
        cur_quat = obs[:, 3:7]
        cur_vel = obs[:, 7:10]
        mask = obs[:, -1]
        
        task_states = obs[:, 10:-1] 
        def_target_pos = task_states[:, 0:3]
        def_target_vel = task_states[:, 30:33]
        def_target_rpy = task_states[:, 60:63]


        task_states_encoded = self.task_encoder(task_states)
        target_pos = def_target_pos + self.encode_lambda * task_states_encoded[:, 0:3]
        target_vel = def_target_vel + self.encode_lambda * task_states_encoded[:, 3:6]
        target_rpy = def_target_rpy + self.encode_lambda * task_states_encoded[:, 6:9]

        step_dt = 0.04

        return self.computeControl(
            step_dt, cur_pos, cur_quat, cur_vel,
            target_pos, target_rpy, target_vel, mask
        )

    def _forward_offline_trajectory(self, obs):
        B, total_dim = obs.shape

        # ---------------------------------------------------
        # 1. Slice Inputs
        # ---------------------------------------------------
        cur_pos  = obs[:, 0:3]
        cur_quat = obs[:, 3:7]
        cur_vel  = obs[:, 7:10]

        # Precomputed integrals (assumed UNCLAMPED raw sums)
        pos_int_state = obs[:, -6:-3]   # (B,3)
        rpy_int_state = obs[:, -3:]     # (B,3)

        # Task / Targets
        task_obs = obs[:, 10:-6]
        def_target_pos = task_obs[:, 0:3]
        def_target_vel = task_obs[:, 30:33]
        def_target_rpy = task_obs[:, 60:63]
        task_states_encoded = self.task_encoder(task_obs)        
        target_pos = def_target_pos + self.encode_lambda * task_states_encoded[:, 0:3]
        target_vel = def_target_vel + self.encode_lambda * task_states_encoded[:, 3:6]
        target_rpy = def_target_rpy + self.encode_lambda * task_states_encoded[:, 6:9]



        # ---------------------------------------------------
        # 2. Geometry
        # ---------------------------------------------------
        cur_rot = quaternion_to_matrix(cur_quat)  # (B,3,3)

        # ---------------------------------------------------
        # 3. POSITION CONTROL (match _dslPIDPositionControl)
        # ---------------------------------------------------
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel

        # --- Integral clamping (EXACTLY same as online) ---
        integral_term = torch.clamp(pos_int_state, -2.0, 2.0)

        int_xy = integral_term[:, :2]
        int_z  = torch.clamp(integral_term[:, 2:3], -0.15, 0.15)
        integral_term = torch.cat([int_xy, int_z], dim=1)

        # PID Force
        target_force = (
            self.P_COEFF_FOR * pos_e
            + self.I_COEFF_FOR * integral_term
            + self.D_COEFF_FOR * vel_e
        )
        target_force[:, 2] += self.GRAVITY

        # Thrust projection
        body_z = cur_rot[:, :, 2]
        scalar_thrust = torch.sum(target_force * body_z, dim=1)
        scalar_thrust = torch.clamp(scalar_thrust, min=0.0)

        Ft = (2 * scalar_thrust) / (1.9 * self.gravity[2]) - 1

        # ---------------------------------------------------
        # 4. Desired Orientation (same as online)
        # ---------------------------------------------------
        target_z_ax = target_force / (torch.norm(target_force, dim=1, keepdim=True) + 1e-6)

        yaw = target_rpy[:, 2]
        target_x_c = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=1)

        cross_z_xc = torch.cross(target_z_ax, target_x_c, dim=1)
        target_y_ax = cross_z_xc / (torch.norm(cross_z_xc, dim=1, keepdim=True) + 1e-6)
        target_x_ax = torch.cross(target_y_ax, target_z_ax, dim=1)

        target_rot = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=2)

        # ---------------------------------------------------
        # 5. ATTITUDE CONTROL (match _dslPIDAttitudeControl)
        # ---------------------------------------------------
        RtR = torch.matmul(target_rot.transpose(1, 2), cur_rot)
        RRT = torch.matmul(cur_rot.transpose(1, 2), target_rot)
        rot_matrix_e = RtR - RRT
        rot_e = torch.stack([rot_matrix_e[:, 2, 1],
                            rot_matrix_e[:, 0, 2],
                            rot_matrix_e[:, 1, 0]], dim=1)

        # D-term: offline assumes zero rates (same assumption as your current code)
        rpy_rates_e = 0.0

        # --- Integral clamping (EXACTLY same as online) ---
        integral_term = torch.clamp(rpy_int_state, -1500.0, 1500.0)

        int_xy = torch.clamp(integral_term[:, :2], -1.0, 1.0)
        int_z  = torch.clamp(integral_term[:, 2:3], -1500.0, 1500.0)
        integral_term = torch.cat([int_xy, int_z], dim=1)

        # PID Torques
        target_torques = (
            -self.P_COEFF_TOR * rot_e
            + self.D_COEFF_TOR * rpy_rates_e
            + self.I_COEFF_TOR * integral_term
        )

        target_torques = torch.clamp(target_torques, -3200.0, 3200.0)

        output_torques = torch.tanh(target_torques / self.moment_scale)

        # ---------------------------------------------------
        # 6. Output
        # ---------------------------------------------------
        return torch.cat([Ft.unsqueeze(1), output_torques], dim=1)

    def preprocess_obs_offline(self, inputs):
        """
        Vectorized processing of trajectory batches.
        
        Args:
            inputs: (B, T, 101) tensor 
        Returns:
            processed_obs: (B, 106) tensor for the final timestep
        """
        B, T, _ = inputs.shape
        device = inputs.device
        step_dt = 0.04

        # 1. Reshape to (B*T, D) to process all timesteps in parallel
        flat_inputs = inputs.view(B * T, -1)
        
        cur_pos = flat_inputs[:, 0:3]
        cur_quat = flat_inputs[:, 3:7]
        cur_vel = flat_inputs[:, 7:10]
        task_obs = flat_inputs[:, 10:-1]
        
        def_target_pos = task_obs[:, 0:3]
        def_target_vel = task_obs[:, 30:33]
        def_target_rpy_cmd = task_obs[:, 60:63]
        task_states_encoded = self.task_encoder(task_obs)        
        target_pos = def_target_pos + self.encode_lambda * task_states_encoded[:, 0:3]
        target_vel = def_target_vel + self.encode_lambda * task_states_encoded[:, 3:6]
        target_rpy_cmd = def_target_rpy_cmd + self.encode_lambda * task_states_encoded[:, 6:9]



        # 2. Position Control (Vectorized across B and T)
        # Note: We pass zero integral state because we are calculating the 
        # instantaneous rotation error to be integrated later.
        zero_int = torch.zeros((B * T, 3), device=device)
        _, target_euler, _, _, _ = self._dslPIDPositionControl(
            step_dt, cur_pos, cur_quat, cur_vel, 
            target_pos, target_rpy_cmd, target_vel, 
            current_integral_state=zero_int
        )

        # 3. Compute Rotation Error (Vectorized)
        # We need the rotation error at every timestep to compute the integral
        rot_e = self._compute_batch_rot_error(cur_quat, target_euler) # (B*T, 3)
        
        # 4. Reshape back to (B, T, 3) for temporal accumulation
        all_pos_e = (target_pos - cur_pos).view(B, T, 3)
        all_rot_e = rot_e.view(B, T, 3)

        # 5. Temporal Integration (Summation over T dimension)
        if self.integral_horizon is not None:
            h = self.integral_horizon
            start_idx = max(0, T - h)
            pos_int_term = torch.sum(all_pos_e[:, start_idx:, :], dim=1) * step_dt
            rpy_int_term = torch.sum(-all_rot_e[:, start_idx:, :], dim=1) * step_dt
        else:
            pos_int_term = torch.sum(all_pos_e, dim=1) * step_dt
            rpy_int_term = torch.sum(-all_rot_e, dim=1) * step_dt

        # 6. Final Clamping (Anti-Windup)
        # pos_int_term = torch.clamp(pos_int_term, -2.0, 2.0)
        # pos_int_term[:, 2] = torch.clamp(pos_int_term[:, 2], -0.15, 0.15)
        
        # rpy_int_term = torch.clamp(rpy_int_term, -1500.0, 1500.0)
        # rpy_int_term[:, :2] = torch.clamp(rpy_int_term[:, :2], -1.0, 1.0)
        pos_xy = pos_int_term[:, :2]
        pos_z  = torch.clamp(pos_int_term[:, 2:3], -0.15, 0.15)
        pos_int_term = torch.cat([pos_xy, pos_z], dim=1)

        rpy_xy = torch.clamp(rpy_int_term[:, :2], -1.0, 1.0)
        rpy_z  = rpy_int_term[:, 2:3]
        rpy_int_term = torch.cat([rpy_xy, rpy_z], dim=1)
        # 7. Construct Output for the last timestep
        # [State(100), Pos_Int(3), RPY_Int(3)]
        last_obs_base = inputs[:, -1, :-1].squeeze(1)
        return torch.cat([last_obs_base, pos_int_term, rpy_int_term], dim=-1)

    def _compute_batch_rot_error(self, quats, target_eulers):
        """Helper to compute rotation error for a flat batch (N, D)."""
        N = quats.shape[0]
        w, x, y, z = quats.unbind(dim=1)
        
        # Current Rotation Matrix
        cur_rot = torch.stack([
            1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y**2 + w*x), # Fixed small index error here
            2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)
        ], dim=1).reshape(N, 3, 3)

        # Target Rotation Matrix
        cr, cp, cy = torch.cos(target_eulers).unbind(dim=1)
        sr, sp, sy = torch.sin(target_eulers).unbind(dim=1)
        target_rot = torch.stack([
            cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr,
            sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr,
            -sp,   cp*sr,             cp*cr
        ], dim=1).reshape(N, 3, 3)

        RtR = torch.matmul(target_rot.transpose(1, 2), cur_rot)
        RRT = torch.matmul(cur_rot.transpose(1, 2), target_rot)
        rot_matrix_e = RtR - RRT
        return torch.stack([rot_matrix_e[:, 2, 1], rot_matrix_e[:, 0, 2], rot_matrix_e[:, 1, 0]], dim=1)