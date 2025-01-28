import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

import imageio
from jumpy.lax import switch
from pygame.examples.music_drop_fade import starting_pos
from stable_baselines3.common.utils import obs_as_tensor
from torchgen.native_function_generation import self_to_out_signature
from wandb.wandb_agent import agent

from toy_envs.bfs_search import map_array
from toy_envs.toy_env_utils import update_location_lava, render_map_and_agent, bfs_shortest_path, masking_obs_lava
import os
import copy
from toy_examples.map import R, L, S, O, W, G, B, A
from toy_examples.map import DOWN, RIGHT, UP, LEFT, STAY


class LavaNavigationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, map_array, agent_pos, render_mode=None, episode_length=100, full_observability=False):
        self.maps = map_array
        # self.initial_map, self._starting_pos, self.goal_pos, self.switch_pos = self._choose_initial_state()
        # self.map = copy.deepcopy(self.initial_map)
        self.grid_size = self.maps[0].shape  # The size of the square grid
        self.visited_map = np.zeros(self.grid_size)
        self.observation_space = spaces.Box(low=L, high=A, shape=(self.grid_size[0], self.grid_size[1]), dtype=np.int64)
        self._starting_pos = agent_pos
        self.action_space = spaces.Discrete(4)  # eliminate stay
        self.observability = full_observability
        # self.num_steps = 0
        self.episode_length = episode_length
        # self.is_goal = False
        # self.is_lava = False
        self.epi_cnt = 0
        # self._agent_pos = np.copy(self._starting_pos)
        # self._new_agent_pos = self._agent_pos
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    def reset(self, seed=10, map_idx=None):
        """
        This is a deterministic environment, so we don't use the seed."""
        self.num_steps = 0
        self.is_goal = False
        self.is_lava = False
        self.switch_on = False
        self.initial_map, self.goal_pos, self.switch_pos = self._choose_initial_state(map_idx)
        self.map = copy.deepcopy(self.initial_map)
        self.map[self._starting_pos[0], self._starting_pos[1]] = A
        self.full_observability = self.observability
        self._agent_pos = np.copy(self._starting_pos)
        self._new_agent_pos = self._agent_pos
        self.epi_cnt += 1
        obs = self._get_obs()
        self.masked_obs = self._masked_observation(obs)
        return obs, self._get_info()

    def step(self, action):
        # print("after changing attiribute:",self._agent_pos)

        self._new_agent_pos, self.map = update_location_lava(agent_pos=self._agent_pos, action=action,
                                                                      map_array=self.map, goal=self.goal_pos, initial_map=self.initial_map)

        if self.initial_map[self._new_agent_pos[0], self._new_agent_pos[1]] == G:
            self.is_goal = True
        if self.initial_map[self._new_agent_pos[0], self._new_agent_pos[1]] == L:
            self.is_lava = True
        if self.initial_map[self._new_agent_pos[0], self._new_agent_pos[1]] == S:
            self.switch_on = True
        observation = self._get_obs()
        terminated = self._get_done()
        info = self._get_info()
        reward = self._get_reward()

        self._agent_pos = self._new_agent_pos
        self.visited_map[self._agent_pos[0], self._agent_pos[1]] = 1
        self.num_steps += 1
        self.masked_obs = self._masked_observation(observation)
        # print("action",action)
        # print(observation)
        return observation, reward, terminated, False, info

    def _get_obs(self):
        # map = self.map.copy()
        # map[self._agent_pos[0], self._agent_pos[1]] = A
        # TODO make observation for partial observable student
        if (self._new_agent_pos == self.switch_pos).all():
            self.full_observability = True
        else:
            self.full_observability = False
            self.map[self.switch_pos[0], self.switch_pos[1]] = S

        if self.full_observability:
            # obs = self.map.reshape(1, self.grid_size[0], self.grid_size[1])
            obs = self.map
        else:
            lavas = np.where(self.map == L)
            obs = copy.deepcopy(self.map)
            obs[lavas] = O
            # obs = obs.reshape(1, self.grid_size[0], self.grid_size[1])

        # if self.is_goal:
        #     terminal_map = np.ones_like(self.map) * -1
        #     terminal_map[self._new_agent_pos[0], self._new_agent_pos[1]] = A
        #     terminal_map[self.goal_pos[0], self.goal_pos[1]] = R
        #     # print(terminal_map)
        #     obs = terminal_map
            # return self._agent_pos
        # print(self.map)
        # else:
        #     obs = self.map

        # feasibility = self._check_obs_feasibility(obs)
        # if feasibility:
        #     return obs
        # else:
        #     print("obs:\n", obs)
        #     print("map\n", self.map)
        #     return None

        return obs

    def _get_done(self):
        # if self.initial_map[self._new_agent_pos[0], self._new_agent_pos[1]] == L:
        #     self.is_lava = True

        terminated = self.num_steps >= self.episode_length or self.is_goal
        return terminated

    def _get_info(self):
        return {"step": self.num_steps, "goal": self.is_goal}

    def _get_reward(self):
        # target_pos = np.argwhere(self.map == R)[0]
        # l1_norm_to_goal = np.linalg.norm(target_pos - self.goal_pos, ord=1)
        # l1_norm_to_target = np.linalg.norm(target_pos - self._new_agent_pos, ord=1)
        # l1_norm = l1_norm_to_goal + l1_norm_to_target

        if self.is_goal:
            return 100 - 90 * (self.num_steps / self.episode_length)
        elif self.is_lava:
            return -10 #-1
        else:
            if self.visited_map[self._new_agent_pos[0], self._new_agent_pos[1]] == 0:
                return 0
            else:
                return -0.5

        # if self.is_goal:
        #     return 1 - 0.9 * (self.num_steps / self.episode_length)
        # elif self.is_lava:
        #     return -1
        # else:
        #     return 0

    def _set_initial_states(self, new_initial_states) -> None:
        # Note: this value should be used only at the next reset
        flat_new_initial_states = [item for sublist in new_initial_states for item in sublist]
        self.initial_states = flat_new_initial_states
        # print("renew initial states", len(self.initial_states))
        # print(self.initial_states.shape)
    def _masked_observation(self, obs):
        if self.switch_on:
            return obs
        else:
            return masking_obs_lava(obs)
    def _choose_initial_state(self, map_idx=None):
        if map_idx is None:
            initial_map = copy.deepcopy(random.choice(self.maps))
        else:
            initial_map = self.maps[map_idx]

        goal_pos = np.argwhere(initial_map == G)[0]
        switch_pos = np.argwhere(initial_map == S)[0]
        # print(initial_map)
        # print("agent:", agent_pos)
        # print("goal :", goal_pos)
        # print("switch:", switch_pos)
        return initial_map, goal_pos, switch_pos



    def _check_obs_feasibility(self, obs):
        agent_pos = np.argwhere(obs == A)
        if len(agent_pos) == 0:
            return False
        else:
            return True

    def render(self):
        """
        Render the environment as an RGB image.

        The agent is represented by a yellow square, empty cells are white, and holes are blue."""
        if self.render_mode == "rgb_array":
            return render_map_and_agent(self.map, self._agent_pos)
        else:
            return
class LavaDie(LavaNavigationEnv):
    def _get_done(self):
        # if self.initial_map[self._new_agent_pos[0], self._new_agent_pos[1]] == L:
        #     self.is_lava = True

        terminated = self.num_steps >= self.episode_length or self.is_goal or self.is_lava
        return terminated

    def _get_info(self):
        return {"step": self.num_steps, "goal": self.is_goal}

    def _get_reward(self):
        # target_pos = np.argwhere(self.map == R)[0]
        # l1_norm_to_goal = np.linalg.norm(target_pos - self.goal_pos, ord=1)
        # l1_norm_to_target = np.linalg.norm(target_pos - self._new_agent_pos, ord=1)
        # l1_norm = l1_norm_to_goal + l1_norm_to_target

        if self.is_goal:
            return 100
        elif self.is_lava:
            return -100
        else:
            if self.visited_map[self._new_agent_pos[0], self._new_agent_pos[1]] == 0:
                return 0
            else:
                return -0.5
class LavaCnn(LavaNavigationEnv):
    def __init__(self, map_array, agent_pos, render_mode=None, episode_length=100, full_observability=False):
        self.maps = map_array
        # self.initial_map, self._starting_pos, self.goal_pos, self.switch_pos = self._choose_initial_state()
        # self.map = copy.deepcopy(self.initial_map)
        self.grid_size = self.maps[0].shape  # The size of the square grid
        self.visited_map = np.zeros(self.grid_size)
        self.observation_space = spaces.Box(low=L, high=A, shape=(1, self.grid_size[0], self.grid_size[1]), dtype=np.int64)
        self._starting_pos = agent_pos
        self.action_space = spaces.Discrete(4)  # eliminate stay
        self.observability = full_observability
        # self.num_steps = 0
        self.episode_length = episode_length
        # self.is_goal = False
        # self.is_lava = False
        self.epi_cnt = 0
        # self._agent_pos = np.copy(self._starting_pos)
        # self._new_agent_pos = self._agent_pos
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    def _get_obs(self):
        # map = self.map.copy()
        # map[self._agent_pos[0], self._agent_pos[1]] = A
        # TODO make observation for partial observable student
        if (self._new_agent_pos == self.switch_pos).all():
            self.full_observability = True
        else:
            self.full_observability = False
            self.map[self.switch_pos[0], self.switch_pos[1]] = S

        if self.full_observability:
            obs = self.map.reshape(1, self.grid_size[0], self.grid_size[1])
        else:
            lavas = np.where(self.map == L)
            obs = copy.deepcopy(self.map)
            obs[lavas] = O
            obs = obs.reshape(1, self.grid_size[0], self.grid_size[1])
        return obs
