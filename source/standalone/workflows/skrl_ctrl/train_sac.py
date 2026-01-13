import torch
import torch.nn as nn
import torch.nn.functional as F


# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from omni.isaac.lab.utils.dict import print_dict
from networks.actor import DiagGaussianActor, DiagGaussianActorPolicy
import os
import gymnasium as gym
# import gym
import numpy as np


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed
# define hidden dimension
actor_hidden_dim = 512
actor_hidden_depth = 3

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

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, hidden_activation=nn.ELU(inplace=True), output_mod=None):
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


# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


GRAVITY = 9.81


class DifferentiableMellinger(nn.Module):
    cf_mass = 0.027
    massThrust = 132000
    INT16_MAX = 65536
    PWM2RPM_SCALE = 0.2685
    PWM2RPM_CONST = 4070.3
    MIN_PWM = 20000
    MAX_PWM = 65535
    GRAVITY = 9.81 * cf_mass
    KF = 3.16e-10

    def __init__(self, max_rpm=21600, ctrl_freq : int = 240, output = "pwm"):
        """

        Args:
            max_rpm: maximum of rpm it can achieve, useless of we output pwm
            ctrl_freq: Control frequency in Hz
            output:     "pwm" or "rpm",
        """
        super().__init__()
        self.CTRL_FREQ = ctrl_freq
        self.MAX_RPM = max_rpm
        self.integral_error = torch.zeros([3, ])

        self.kp_xy = torch.nn.Parameter(torch.tensor(0.4, requires_grad=True))
        self.ki_xy = torch.nn.Parameter(torch.tensor(0.05, requires_grad=False))
        self.kd_xy = torch.nn.Parameter(torch.tensor(0.2, requires_grad=False))
        # Z position
        self.kp_z = torch.nn.Parameter(torch.tensor(1.25, requires_grad=False))
        self.ki_z = torch.nn.Parameter(torch.tensor(0.05, requires_grad=False))
        self.kd_z = torch.nn.Parameter(torch.tensor(0.4, requires_grad=False))
        # Attitude
        self.kR_xy = torch.nn.Parameter(torch.tensor(70000., requires_grad=True))
        self.kw_xy = torch.nn.Parameter(torch.tensor(0., requires_grad=False))
        self.ki_m_xy = torch.nn.Parameter(torch.tensor(20000., requires_grad=False))

        self.kR_z = torch.nn.Parameter(torch.tensor(60000., requires_grad=False))
        self.kw_z = torch.nn.Parameter(torch.tensor(500., requires_grad=False))
        self.ki_m_z = torch.nn.Parameter(torch.tensor(12000., requires_grad=False))

        # # XY positions
        # self.kp_xy = 0.4
        # self.ki_xy = 0.05
        # self.kd_xy = 0.2
        # # Z position
        # self.kp_z = 1.25
        # self.ki_z = 0.05
        # self.kd_z = 0.4
        # # Attitude
        # self.kR_xy = 70000.
        # self.kw_xy = 0.
        # self.ki_m_xy = 20000.
        # # Z Altitude
        # self.kR_z = 60000.
        # self.kw_z = 500.
        # self.ki_m_z = 12000.
        #
        # self.P_COEFF_FOR = torch.nn.Parameter(torch.tensor([self.kp_xy, self.kp_xy, self.kp_z], requires_grad=True))
        # self.I_COEFF_FOR = torch.nn.Parameter(torch.tensor([self.ki_xy, self.ki_xy, self.ki_z], requires_grad=False))
        # self.D_COEFF_FOR = torch.nn.Parameter(torch.tensor([self.kd_xy, self.kd_xy, self.kd_z], requires_grad=False))
        # self.P_COEFF_TOR = torch.nn.Parameter(torch.tensor([self.kR_xy, self.kR_xy, self.kR_z], requires_grad=True))
        # self.I_COEFF_TOR = torch.nn.Parameter(torch.tensor([self.kw_xy, self.kw_xy, self.kw_z], requires_grad=False))
        # self.D_COEFF_TOR = torch.nn.Parameter(torch.tensor([self.ki_m_xy, self.ki_m_xy, self.ki_m_z], requires_grad=False))

        self.i_range_xy = 2.0
        self.i_range_z = 0.4
        self.i_range_m_xy = 1.0
        self.i_range_m_z = 1500.

        self.target_rpy_rates = torch.zeros([3,]).float().to(torch.device('cuda'))
        self.MIXER_MATRIX = torch.tensor([
            [-.5, -.5, -1],
            [-.5, .5, 1],
            [.5, .5, -1],
            [.5, -.5, 1]
        ]).float()
        self.goal = torch.tensor([0., 0., 1.]).to(device)
        self.target_x_c = torch.tensor([1., 0., 0.]).to(device) # assume target rpy always 0
        self.gravity = torch.tensor([0, 0, self.GRAVITY]).to(device)
        self.output_type = output
        self.reset()

        self.trunk = mlp(28, 256, 4,
                              2, hidden_activation=nn.ELU(inplace=True))

        self.log_std_bounds=[-20., 1.]

        def transpose(x):
            return x.T
        self.vec_transpose = torch.vmap(transpose)

    def projection_on_gains(self):
        with torch.no_grad():
            self.kp_xy.clamp_(min=0.1)
            self.kR_xy.clamp_(min=10000.)

    def get_controller_parameters_dict(self):

        return self.state_dict()


    def set_device(self, device):
        self.MIXER_MATRIX = self.MIXER_MATRIX.to(device)
        self.goal = self.goal.to(device)
        self.target_x_c = self.target_x_c.to(device)
        self.gravity = self.gravity.to(device)


    def set_env_params(self, thrust_to_weight: float, robot_weight: float, moment_scale: float):
        """Provide environment-specific scaling parameters so the controller
        can convert physical outputs (thrust, torques) to the environment's
        normalized action space.

        Args:
            thrust_to_weight: cfg.thrust_to_weight from the env
            robot_weight: computed robot weight (self._robot_weight in env)
            moment_scale: cfg.moment_scale from the env
        """
        # store as tensors on same device as MIXER_MATRIX
        dev = self.MIXER_MATRIX.device
        self.thrust_to_weight = torch.tensor(thrust_to_weight, device=dev, dtype=self.MIXER_MATRIX.dtype)
        self.robot_weight = torch.tensor(robot_weight, device=dev, dtype=self.MIXER_MATRIX.dtype)
        self.moment_scale = torch.tensor(moment_scale, device=dev, dtype=self.MIXER_MATRIX.dtype)


    def physical_to_normalized_action(self, control: torch.Tensor) -> torch.Tensor:
        """Convert controller physical outputs into env-normalized actions in [-1,1].

        Expects `control` shape (batch,4) where control[:,0] is a thrust-related
        scalar (we assume this is proportional to scalar_thrust computed in the
        controller) and control[:,1:4] are target torques in physical units.

        Returns a tensor of same shape with values clipped to [-1,1].
        """
        # control shape check
        if control.dim() == 1:
            control = control.unsqueeze(0)

        # If env params not set, return control unchanged (caller should ensure params set)
        if not hasattr(self, "thrust_to_weight") or not hasattr(self, "robot_weight") or not hasattr(self, "moment_scale"):
            return control

        # Infer whether the controller returned PWM-scaled thrust (massThrust * scalar_thrust)
        # or scalar_thrust directly. We prefer to use scalar_thrust for mapping to env.
        # If massThrust is present and non-zero, divide by it to recover scalar_thrust.
        mass_thrust = getattr(self, "massThrust", None)
        thrust_raw = control[:, 0]
        if mass_thrust is not None and mass_thrust != 0:
            # ensure tensor type
            try:
                mt = float(mass_thrust)
            except Exception:
                mt = mass_thrust
            # avoid division by zero
            if mt != 0:
                desired_thrust = thrust_raw / mt
            else:
                desired_thrust = thrust_raw
        else:
            desired_thrust = thrust_raw

        # map desired_thrust to env action0 using env mapping:
        # env_thrust = thrust_to_weight * robot_weight * (action0 + 1)/2
        denom = (self.thrust_to_weight * self.robot_weight)
        denom = torch.clamp(denom, min=1e-8)
        action0 = 2.0 * (desired_thrust / denom) - 1.0

        # map torques: env uses moment_scale * action[1:]
        torques = control[:, 1:4]
        action_mom = torques / self.moment_scale

        actions = torch.cat([action0.unsqueeze(1), action_mom], dim=1)
        actions = torch.clamp(actions, -1.0, 1.0)
        return actions


    def quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part last,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        i, j, k, r = torch.unbind(quaternions, -1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
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

    def _index_from_letter(self, letter: str) -> int:
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2
        raise ValueError("letter must be either X, Y or Z.")

    def _angle_from_tan(self,
            axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
    ) -> torch.Tensor:
        """
        Extract the first or third Euler angle from the two members of
        the matrix which are positive constant times its sine and cosine.

        Args:
            axis: Axis label "X" or "Y or "Z" for the angle we are finding.
            other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
                convention.
            data: Rotation matrices as tensor of shape (..., 3, 3).
            horizontal: Whether we are looking for the angle for the third axis,
                which means the relevant entries are in the same row of the
                rotation matrix. If not, they are in the same column.
            tait_bryan: Whether the first and third axes in the convention differ.

        Returns:
            Euler Angles in radians for each matrix in data as a tensor
            of shape (...).
        """

        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return torch.atan2(data[..., i1], data[..., i2])
        if tait_bryan:
            return torch.atan2(-data[..., i2], data[..., i1])
        return torch.atan2(data[..., i2], -data[..., i1])

    def matrix_to_euler_angles(self, matrix: torch.Tensor, convention = "XYZ") -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to Euler angles in radians.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).
            convention: Convention string of three uppercase letters.

        Returns:
            Euler angles in radians as tensor of shape (..., 3).
        """
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        i0 = self._index_from_letter(convention[0])
        i2 = self._index_from_letter(convention[2])
        tait_bryan = i0 != i2
        if tait_bryan:
            central_angle = torch.asin(
                matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
            )
        else:
            central_angle = torch.acos(matrix[..., i0, i0])

        o = (
            self._angle_from_tan(
                convention[0], convention[1], matrix[..., i2], False, tait_bryan
            ),
            central_angle,
            self._angle_from_tan(
                convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
            ),
        )
        return torch.stack(o, -1)


    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = torch.zeros(3).to(device)
        #### Initialized PID control variables #####################
        self.last_pos_e = torch.zeros(1, 3).to(device)
        self.integral_pos_e = torch.zeros(1, 3).to(device)
        self.last_rpy_e = torch.zeros(3).to(device)
        self.integral_rpy_e = torch.zeros(3).to(device)

    def mellinger_control(self, obs):
        #### OBS SPACE OF SIZE 28
        # first 16: xyz_error 3, quat 4, rpy 3, vel_xyz 3, angle_vel_xyz 3 each
        #### then 12: integral error of pos 3, diff error of pos 3, integral error of angle 3, diff error of angle 3

        P_COEFF_FOR = torch.stack([self.kp_xy, self.kp_xy, self.kp_z])
        I_COEFF_FOR = torch.stack([self.ki_xy, self.ki_xy, self.ki_z])
        D_COEFF_FOR = torch.stack([self.kd_xy, self.kd_xy, self.kd_z])
        P_COEFF_TOR = torch.stack([self.kR_xy, self.kR_xy, self.kR_z])
        I_COEFF_TOR = torch.stack([self.kw_xy, self.kw_xy, self.kw_z])
        D_COEFF_TOR = torch.stack([self.ki_m_xy, self.ki_m_xy, self.ki_m_z])

        # position control
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        pos_e = self.goal - obs[:, 0:3]
        cur_quat = obs[:, 3:7]
        cur_rpy = obs[:, 7:10]
        cur_vel = obs[:, 10:13]
        integral_pos_error = obs[:, 16:19]
        diff_pos_error = obs[:, 19:22]
        integral_rpy_error = obs[:, 22:25]
        diff_rpy_error = -1 * obs[:, 25:28]
        cur_rotation = self.quaternion_to_matrix(cur_quat)
        vel_e = - cur_vel
        # self.integral_pos_e = self.integral_pos_e + pos_e * 1 / 240
        # self.integral_pos_e = torch.clip(self.integral_pos_e, -2., 2.)
        # self.integral_pos_e[:, 2] = torch.clip(self.integral_pos_e[:, 2], -0.15, .15)
        #### PID target thrust #####################################
        target_thrust = torch.multiply(P_COEFF_FOR, pos_e) \
                        + torch.multiply(I_COEFF_FOR, integral_pos_error) \
                        + torch.multiply(D_COEFF_FOR, vel_e) + self.gravity  # , device=self.de
        scalar_thrust = torch.clamp(torch.vmap(torch.inner)(target_thrust, cur_rotation[:, :, 2]), 0, torch.inf)
        # thrust_pwm = (torch.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        thrust_pwm = self.massThrust * scalar_thrust
        target_z_ax = F.normalize(target_thrust, dim=1)
        # target_x_c = torch.tensor([1., 0., 0.]) # assume target rpy always 0
        target_y_ax = F.normalize(torch.vmap(torch.cross, in_dims=(0, None))(target_z_ax, self.target_x_c),
                                  dim=1)  # / torch.norm(torch.cross(target_z_ax, target_x_c))
        target_x_ax = torch.vmap(torch.cross)(target_y_ax, target_z_ax)
        target_rotation_transposed = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=1)
        target_rotation = torch.permute(target_rotation_transposed, [0, 2, 1])
        #### Target rotation #######################################
        target_euler = self.matrix_to_euler_angles(target_rotation)

        # Altitude control
        # cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        # cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        # target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        # w, x, y, z = target_quat
        # target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()

        rot_matrix_e = (torch.matmul(self.vec_transpose(target_rotation), cur_rotation)
                        - torch.matmul(self.vec_transpose(cur_rotation), target_rotation))
        rot_e = torch.stack([rot_matrix_e[:, 2, 1], rot_matrix_e[:, 0, 2], rot_matrix_e[:, 1, 0]], dim=1)
        self.rot_e = rot_e
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e / 240
        self.integral_rpy_e = torch.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = torch.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - torch.multiply(P_COEFF_TOR, rot_e) \
                         + torch.multiply(D_COEFF_TOR, diff_rpy_error) \
                         + torch.multiply(I_COEFF_TOR, integral_rpy_error)
        target_torques = torch.clip(target_torques, -32000, 32000)
        # pwm = thrust_pwm.unsqueeze(1) + torch.matmul(self.MIXER_MATRIX, target_torques.unsqueeze(2)).squeeze()
        # pwm = torch.clip(pwm, self.MIN_PWM, self.MAX_PWM)  # .squeeze(dim=-1)
        # if self.output_type == "pwm":
        #     return pwm / self.MAX_PWM
        # elif self.output_type == "rpm":
        #     return (self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST) / self.MAX_RPM
        # else:
        #     raise ValueError(f"Invalid output type {self.output_type}.")
        return torch.cat([thrust_pwm.unsqueeze(1), target_torques], dim=1).squeeze()


    def forward(self, obs):
        control = self.mellinger_control(obs)
        log_std = self.trunk(obs)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        return control, log_std, {}
    
# define models (stochastic and deterministic models) using mixins
class StochasticMellingerActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = DifferentiableMellinger()
        self.net.set_device(device)


    def compute(self, inputs, role):
        # DifferentiableMellinger.forward returns (control, log_std, extras).
        # Unpack and convert physical control outputs into environment-normalized
        # actions before returning (mean, log_std, extras) as expected by GaussianMixin.
        mean_raw, log_std, extras = self.net(inputs['states'])

        # If the net has env params, convert physical outputs -> normalized actions.
        if hasattr(self.net, "physical_to_normalized_action"):
            try:
                mean = self.net.physical_to_normalized_action(mean_raw)
            except Exception:
                # Fallback to raw output if conversion fails for any reason
                mean = mean_raw
        else:
            mean = mean_raw

        return mean, log_std - log_std, extras



class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# load and wrap the Isaac Lab environment
cli_args = ["--video", "--enable_cameras"]  # enable video recording and camera rendering
# load and wrap the Isaac Gym environment
task_version = "legtrain"
finetune = task_version == 'legtrain-finetune'
task_name = f"Isaac-Quadcopter-{task_version}-Trajectory-Direct-v0"
num_envs = 32 if not finetune else 1
env = load_isaaclab_env(task_name=task_name, num_envs=num_envs, cli_args=cli_args)

video_kwargs = {
    "video_folder": os.path.join(f"runs/torch/{task_version}/", "videos", "train", "SAC"),
    "step_trigger": lambda step: step % 10000== 0,
    "video_length": 400,
    "disable_logger": True,
}
print("[INFO] Recording videos during training.")
print_dict(video_kwargs, nesting=4)
env = gym.wrappers.RecordVideo(env, **video_kwargs)

env = wrap_env(env)


device = env.device



# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=int(1e5), num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = StochasticMellingerActor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# Wire environment scaling parameters into the policy's differentiable controller
# so it can convert its physical outputs into the env's normalized action space.
try:
    unwrapped = getattr(env, "unwrapped", env)
    thrust_to_weight = getattr(getattr(unwrapped, "cfg", None), "thrust_to_weight", None)
    robot_weight = getattr(unwrapped, "_robot_weight", None)
    moment_scale = getattr(getattr(unwrapped, "cfg", None), "moment_scale", None)
    if thrust_to_weight is not None and robot_weight is not None and moment_scale is not None:
        try:
            models["policy"].net.set_env_params(thrust_to_weight, robot_weight, moment_scale)
            print(f"[INFO] Set env params on policy.net: thrust_to_weight={thrust_to_weight}, robot_weight={robot_weight}, moment_scale={moment_scale}")
            # Quick sanity check: estimate the normalized action that corresponds to gravity/hover
            try:
                gravity_z = float(models["policy"].net.gravity[2].to(models["policy"].net.MIXER_MATRIX.device))
                denom = float(models["policy"].net.thrust_to_weight * models["policy"].net.robot_weight)
                if denom != 0:
                    hover_action0 = 2.0 * (gravity_z / denom) - 1.0
                    print(f"[INFO] Estimated hover action0 (before clamp) = {hover_action0:.4f} (should be in [-1,1] for nominal values)")
                else:
                    print("[WARN] denom is zero when computing hover action; skipping hover estimate")
            except Exception as e:
                print("[WARN] Failed to compute hover-action sanity check:", e)
        except Exception as e:
            print("[WARN] Failed to set env params on policy.net:", e)
    else:
        print("[WARN] Env scaling params not found; policy will return physical outputs unless handled elsewhere.")
except Exception as e:
    print("[WARN] Error while wiring env params to policy.net:", e)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 256
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 0.0#1e-4
cfg["critic_learning_rate"] = 1e-4
cfg["random_timesteps"] = 0#12e3 if not finetune else 0
cfg["learning_starts"] = 12e3 if not finetune else 25e3
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True if not finetune else False
cfg["entropy_learning_rate"] = 1e-4
cfg["initial_entropy_value"] = 1.0 if not finetune else 0.06
# logging to wandb and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 10000 if not finetune else 1000
cfg["experiment"]["directory"] = f"runs/torch/{task_name}/SAC"
cfg["experiment"]["experiment_name"] = f"{task_name}-SAC"
cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"] = {
    "project": "Isaac-Lab-Quadcopter-SAC",
    "name": f"{task_name}-SAC",
    "config": {
        "task": task_name,
        "num_envs": num_envs,
        "batch_size": 256,
    }
}


agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
timesteps = int(3e5) if not finetune else int(5e4)
if finetune:
    agent.load("/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-legtrain-Trajectory-Direct-v0/SAC/25-02-01_23-09-09-780539_SAC/checkpoints/agent_300000.pt")
    memory.load("/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-legtrain-Trajectory-Direct-v0/SAC/memories/25-02-11_04-00-39-059963_memory_0x7fb974303e50.pt")

cfg_trainer = {"timesteps": timesteps, "headless": True, "environment_info": "log"}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
memory.save(cfg["experiment"]["directory"], format = 'pt')

# start training
trainer.train()