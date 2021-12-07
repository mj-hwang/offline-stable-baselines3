import io
import pathlib
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


class OfflineAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        policy_base: Type[BasePolicy],
        learning_rate: Union[float, Schedule],
        buffer_size: int = 1_000_000,  # 1e6
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        remove_time_limit_termination: bool = False,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super(OfflineAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage
        self.replay_buffer_class = replay_buffer_class
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        self.replay_buffer_kwargs = replay_buffer_kwargs
        self._episode_storage = None

        # Remove terminations (dones) that are due to time limit
        # see https://github.com/hill-a/stable-baselines/issues/863
        self.remove_time_limit_termination = remove_time_limit_termination


        self.actor = None  # type: Optional[th.nn.Module]
        self.replay_buffer = None  # type: Optional[ReplayBuffer]
        # Update policy keyword arguments
        if sde_support:
            self.policy_kwargs["use_sde"] = self.use_sde
        # For gSDE only
        self.use_sde_at_warmup = use_sde_at_warmup

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Use DictReplayBuffer if needed
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, gym.spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        elif self.replay_buffer_class == HerReplayBuffer:
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"

            # If using offline sampling, we need a classic replay buffer too
            if self.replay_buffer_kwargs.get("online_sampling", True):
                replay_buffer = None
            else:
                replay_buffer = DictReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.action_space,
                    self.device,
                    optimize_memory_usage=self.optimize_memory_usage,
                )

            self.replay_buffer = HerReplayBuffer(
                self.env,
                self.buffer_size,
                self.device,
                replay_buffer=replay_buffer,
                **self.replay_buffer_kwargs,
            )

        if self.replay_buffer is None:
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Save the replay buffer as a pickle file.
        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ) -> None:
        """
        Load a replay buffer from a pickle file.
        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(self.replay_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.replay_buffer.handle_timeout_termination = False
            self.replay_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)

        if isinstance(self.replay_buffer, HerReplayBuffer):
            assert self.env is not None, "You must pass an environment at load time when using `HerReplayBuffer`"
            self.replay_buffer.set_env(self.get_env())
            if truncate_last_traj:
                self.replay_buffer.truncate_last_trajectory()

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        cf `BaseAlgorithm`.
        """
        # Prevent continuity issue by truncating trajectory
        # when using memory efficient replay buffer
        # see https://github.com/DLR-RM/stable-baselines3/issues/46

        # Special case when using HerReplayBuffer,
        # the classic replay buffer is inside it when using offline sampling
        if isinstance(self.replay_buffer, HerReplayBuffer):
            replay_buffer = self.replay_buffer.replay_buffer
        else:
            replay_buffer = self.replay_buffer

        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and replay_buffer is not None
            and (replay_buffer.full or replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            # Go to the previous index
            pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            replay_buffer.dones[pos] = True

        return super()._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            log_path,
            reset_num_timesteps,
            tb_log_name,
        )
    
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OfflineAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        while self._n_updates < total_timesteps:
            self.num_timesteps += self.env.num_envs
            self._on_step()
            if self.num_timesteps > 0:
                self.train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)

        callback.on_training_end()

        return self

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        raise NotImplementedError()

    # def _sample_action(
    #     self,
    #     learning_starts: int,
    #     action_noise: Optional[ActionNoise] = None,
    #     n_envs: int = 1,
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Sample an action according to the exploration policy.
    #     This is either done by sampling the probability distribution of the policy,
    #     or sampling a random action (from a uniform distribution over the action space)
    #     or by adding noise to the deterministic output.
    #     :param action_noise: Action noise that will be used for exploration
    #         Required for deterministic policy (e.g. TD3). This can also be used
    #         in addition to the stochastic policy for SAC.
    #     :param learning_starts: Number of steps before learning for the warm-up phase.
    #     :param n_envs:
    #     :return: action to take in the environment
    #         and scaled action that will be stored in the replay buffer.
    #         The two differs when the action space is not normalized (bounds are not [-1, 1]).
    #     """
    #     # Select an action according to policy
    #     unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

    #     # Rescale the action from [low, high] to [-1, 1]
    #     if isinstance(self.action_space, gym.spaces.Box):
    #         scaled_action = self.policy.scale_action(unscaled_action)

    #         # Add noise to the action (improve exploration)
    #         if action_noise is not None:
    #             scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

    #         # We store the scaled action in the buffer
    #         buffer_action = scaled_action
    #         action = self.policy.unscale_action(scaled_action)
    #     else:
    #         # Discrete case, no need to normalize or clip
    #         buffer_action = unscaled_action
    #         action = buffer_action
    #     return action, buffer_action

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time_elapsed + 1e-8))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _on_step(self) -> None:
        """
        Method called after each step in the environment.
        It is meant to trigger DQN target network update
        but can be used for other purposes
        """
        pass

    # def evaluate(
    #     self,
    #     env: VecEnv,
    #     callback: BaseCallback,
    #     train_freq: TrainFreq,
    #     replay_buffer: ReplayBuffer,
    #     action_noise: Optional[ActionNoise] = None,
    #     learning_starts: int = 0,
    #     log_interval: Optional[int] = None,
    # ) -> RolloutReturn:

    #     # Switch to eval mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(False)

    #     num_collected_steps, num_collected_episodes = 0, 0

    #     assert isinstance(env, VecEnv), "You must pass a VecEnv"
    #     assert train_freq.frequency > 0, "Should at least collect one step or episode."

    #     if env.num_envs > 1:
    #         assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

    #     # Vectorize action noise if needed
    #     if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
    #         action_noise = VectorizedActionNoise(action_noise, env.num_envs)

    #     if self.use_sde:
    #         self.actor.reset_noise(env.num_envs)

    #     callback.on_rollout_start()
    #     continue_training = True

    #     while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
    #         if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
    #             # Sample a new noise matrix
    #             self.actor.reset_noise(env.num_envs)

    #         # Select action randomly or according to policy
    #         actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

    #         # Rescale and perform action
    #         new_obs, rewards, dones, infos = env.step(actions)

    #         self.num_timesteps += env.num_envs
    #         num_collected_steps += 1

    #         # Give access to local variables
    #         callback.update_locals(locals())
    #         # Only stop training if return value is False, not when it is None.
    #         if callback.on_step() is False:
    #             return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

    #         # Retrieve reward and episode length if using Monitor wrapper
    #         self._update_info_buffer(infos, dones)

    #         # Store data in replay buffer (normalized action and unnormalized observation)
    #         self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

    #         self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

    #         # For DQN, check if the target network should be updated
    #         # and update the exploration schedule
    #         # For SAC/TD3, the update is dones as the same time as the gradient update
    #         # see https://github.com/hill-a/stable-baselines/issues/900
    #         self._on_step()

    #         for idx, done in enumerate(dones):
    #             if done:
    #                 # Update stats
    #                 num_collected_episodes += 1
    #                 self._episode_num += 1

    #                 if action_noise is not None:
    #                     kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
    #                     action_noise.reset(**kwargs)

    #                 self._dump_logs()

    #     callback.on_rollout_end()

    #     return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
