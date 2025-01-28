
from policies.BC import BCCustom
from policies.custom_rollout import *
from policies.bfs import BFS
from cfg_utils import *
from omegaconf import DictConfig, OmegaConf
import hydra
from imitation.data.types import Trajectory
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from toy_examples.map import W, O, R, B, G, A
def rollout_steps(model, env, max_steps=None, iseval=False, isStudent=True):
    """
    Perform a single rollout of the environment and return a trajectory.

    Parameters:
        model: Trained model to predict actions.
        env: Environment to interact with.
        max_steps: Optional; maximum number of steps for the rollout.

    Returns:
        A single trajectory containing observations, actions, and metadata.
    """
    global frames
    obs = env.reset()
    if iseval:
        print("initial scence \n", obs)
    if isinstance(obs, tuple):  # Handle VecEnv API
        obs, _ = obs

    trajectory = {"obs": [], "acts": [], "infos": []}
    obs_raw_array = []
    done = False
    step_count = 0
    frames.append(env.render())
    while not done:
        if isStudent:
            masked_observation = masking_obs(obs)
            action, _ = model.predict(masked_observation, deterministic=True)  # Predict action
        else:
            action, _ = model.predict(obs, deterministic=True)  # Predict action
        # action = action  # Convert to scalar if necessary

        # Append current state and action
        # trajectory["obs"].append(obs.copy())
        obs_raw_array.append(obs.copy())
        trajectory["acts"].append(action)

        # Take an environment step

        obs, reward, done, info = env.step(action)
        # callback._on_step().
        if isinstance(obs, tuple):  # Handle VecEnv API
            obs, info = obs

        trajectory["infos"].append(info)
        step_count += 1
        frames.append(env.render())
        # Stop if max_steps is specified and reached
        if max_steps and step_count >= max_steps:
            break

    # Append the final observation
    # trajectory["obs"].append(obs.copy())
    obs_raw_array.append(obs.copy())
    for obs in obs_raw_array:
        masked_observation = masking_obs(obs)
        trajectory["obs"].append(masked_observation)
    if iseval:
        if reward == 1:
            print("goal reached : ", len(trajectory['obs']))

        else:
            print("failed in finding solution")
        print(trajectory["acts"])

    return Trajectory(
        obs=np.array(trajectory["obs"]),
        acts=np.array(trajectory["acts"]),
        infos=trajectory["infos"],
        terminal=True,
    )
def collect_demonstrations(model, env, num_episodes=10, max_steps=None, iseval=False):
    """
    Collect multiple trajectories (demonstrations) using the provided model.

    Parameters:
        model: Trained model to predict actions.
        env: Environment to interact with.
        num_episodes: Number of episodes to collect.
        max_steps: Optional; maximum number of steps per episode.

    Returns:
        A list of Trajectory objects.
    """
    trajectories = []
    for _ in range(num_episodes):
        trajectory = rollout_steps(model, env, max_steps, iseval, isStudent=False)
        trajectories.append(trajectory)

    return trajectories

@hydra.main(version_base=None, config_path="config", config_name="train_ensemble_config")
def train(cfg: DictConfig):
    map_array = load_map_from_example_dict(cfg.env.example_name)
    episode_length = cfg.env.episode_length
    grid_class = GridNavigationEnv
    env = grid_class(map_array=np.copy(map_array), render_mode="rgb_array",
                                 episode_length=episode_length)
    # make_env_fn = lambda: Monitor(
    #     grid_class(map_array=np.copy(map_array), render_mode="rgb_array",
    #                episode_length=episode_length))
    # env = make_vec_env(
    #     make_env_fn,
    #     n_envs=cfg.compute.n_cpu_workers,
    #     seed=cfg.seed,
    #     vec_env_cls=SubprocVecEnv,
    # )
    # env_dual_obs = CustomObservationWrapper(env)
    # env = grid_class(map_array=np.copy(map_array), starting_pos=starting_pos, goal_pos=goal_pos, render_mode="rgb_array",
    #                          episode_length=episode_length)

    # video_callback = GridNavVideoRecorderCallback(
    #     eval_env,
    #     rollout_save_path=os.path.join(cfg.logging.run_path, "eval"),
    #     render_freq=cfg.logging.video_save_freq // cfg.compute.n_cpu_workers,
    #     map_array=np.copy(map_array),
    #     goal_pos=goal_pos,
    # )

    pretrained_model = BFS(env=env)
    # Step 2: Generate Demonstrations

    # Collect demonstrations
    # trajectories = collect_demonstrations(pretrained_model, env)
    # trajectories = generate_unmasked_trajectories(pretrained_model, env, n_episodes=10)
    trajectories = [generate_unmasked_trajectories(pretrained_model, env, n_episodes=10) for _ in range(cfg.bc_algo.n_ensemble)]
    # Step 3: Train an Imitation Learning Agent
    # Flatten environment required for BC training
    # bc_env = DummyVecEnv([lambda: Monitor(GridNavigationEnv(render_mode="rgb_array"))])

    # Initialize Behavior Cloning algorithm
    bc_ensemble = []
    for i in range(cfg.bc_algo.n_ensemble):
        rng = np.random.default_rng(seed=i)
        bc = BCCustom(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=trajectories[i],
            rng=rng,
        )
        # initialize policy
        bc.train(n_epochs=50) # 10 episode 10 ephoch / BC : 10 episode 5000 epoch
        bc_ensemble.append(bc)
    #DAgger
    total_episode = 100
    trajectory = trajectories[0]
    for i in range(total_episode):
        print("#####\nIteration : ",i,"\n####")
        trajectory = collect_augmented_trajectories_ensemble(trajectory, pretrained_model, bc_ensemble, env)
        bc_ensemble, trajectories = update_trajectories(trajectory, trajectories, bc_ensemble, cfg.bc_algo.n_ensemble)
        for bc in bc_ensemble:
            bc.train(n_epochs=10)
    evaluation, _ = evaluate_policy(bc_ensemble[0].policy, env, cfg.log_path, 3)
    model_save_path = os.path.join(cfg.log_path, "imitation_policy.pth")
    model_save_dir = os.path.dirname(model_save_path)
    os.makedirs(model_save_dir, exist_ok=True)

    bc.policy.save(model_save_path)
    print("Imitation learning agent saved!")

if __name__ == "__main__":
    # train_or_sweep()

    train()
