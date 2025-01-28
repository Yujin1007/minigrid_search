from imitation.data.rollout import generate_trajectories
from random import random

from sympy.physics.quantum.identitysearch import bfs_identity_search

from toy_envs.toy_env_utils import *
from torch.backends.cudnn import deterministic

from imitation.data.types import Trajectory
from toy_envs.grid_nav import *

def generate_unmasked_trajectories(expert_policy, env, n_episodes=10):
    """
    Generate trajectories for the expert using unmasked observations.

    Parameters:
        expert_policy: The expert policy.
        env: The DualObservationWrapper environment.
        n_episodes: Number of episodes to collect.

    Returns:
        Trajectories collected by the expert.
    """
    trajectories = []

    for _ in range(n_episodes):
        trajectory = {"obs": [], "acts": [], "infos": []}

        obs, _ = env.reset()
        masked_obs = masking_obs(obs)
        # obs = obs["unmasked"]  # Use unmasked observation for the expert
        done = False
        trajectory["obs"].append(masked_obs)
        while not done:
            action = expert_policy.predict(obs, deterministic=True)
            # print(action)
            # action = action.item()
            obs, reward, done, _, info = env.step(action)
            masked_obs = masking_obs(obs)
            # obs = obs["unmasked"]  # Always use unmasked for the expert
            trajectory["acts"].append(action)
            trajectory["infos"].append(info)
            trajectory["obs"].append(masked_obs)

        trajectories.append(
            Trajectory(
                obs=np.array(trajectory["obs"]),
                acts=np.array(trajectory["acts"]),
                infos=trajectory["infos"],
                terminal=True,
            )
        )

    return trajectories

def collect_augmented_trajectories(trajectories, expert_policy, bc_policy, env, beta=0.8, full_visibility=False):
    """
    Generate trajectories for the expert using unmasked observations.

    Parameters:
        expert_policy: The expert policy.
        env: The DualObservationWrapper environment.
        n_episodes: Number of episodes to collect.

    Returns:
        Trajectories collected by the expert.
    """
    obs, _  = env.reset()
    masked_obs = masking_obs(obs)
    # obs = obs["unmasked"]  # Use unmasked observation for the expert
    done = False


    trajectory = {"obs": [], "acts": [], "infos": []}
    trajectory["obs"].append(masked_obs)

    while not done:
        if full_visibility:
            action_bc = bc_policy.predict(obs, deterministic=True)
        else:
            action_bc = bc_policy.predict(masked_obs, deterministic=True)
        action = expert_policy.predict(obs, deterministic=True)
        if action == 4: # stuck
            print("stuck done")
            break
        if isinstance(action, tuple):  # Handle VecEnv API
            action, _ = action
            action = action.item()
        if isinstance(action_bc, tuple):  # Handle VecEnv API
            action_bc, _ = action_bc
            action_bc = action_bc.item()

        if action_bc != action:
            dumb_env = copy.deepcopy(env)
            obs, _, _, _, info = dumb_env.step(action)
            _, reward, done, _, _ = env.step(action_bc)
        else:
            obs, reward, done, _, info = env.step(action)
        masked_obs = masking_obs(obs)
        # obs = obs["unmasked"]  # Always use unmasked for the expert
        trajectory["acts"].append(action)
        trajectory["infos"].append(info)
        if full_visibility:
            trajectory["obs"].append(obs.copy())
        else:
            trajectory["obs"].append(masked_obs.copy())
    if len(trajectory["acts"])>0:
        trajectories.append(
            Trajectory(
                obs=np.array(trajectory["obs"]),
                acts=np.array(trajectory["acts"]),
                infos=trajectory["infos"],
                terminal=True,
            )
        )
    return trajectories
def collect_augmented_trajectories2(trajectories, expert_policy, bc_policy, env, full_visibility=False, task_level=1):
    """
    Generate trajectories for the expert using unmasked observations.

    Parameters:
        expert_policy: The expert policy.
        env: The DualObservationWrapper environment.
        n_episodes: Number of episodes to collect.

    Returns:
        Trajectories collected by the expert.
    """
    obs, _  = env.reset()
    masked_obs = masking_obs(obs)
    # obs = obs["unmasked"]  # Use unmasked observation for the expert
    done = False


    trajectory = {"obs": [], "acts": [], "infos": []}
    trajectory["obs"].append(masked_obs)
    bfs_path = []
    itr = 0
    frames = [env.render()]
    blue_at_goal = 0
    while not done:
        if full_visibility:
            action = bc_policy.predict(obs,deterministic=True)
        else:
            action = bc_policy.predict(masked_obs,deterministic=True)

        if isinstance(action, tuple):  # Handle VecEnv API
            action, _ = action
            action = action.item()

        blue_poses = np.argwhere(obs == B)

        if any(np.array_equal(blue_pos, env.goal_pos) for blue_pos in blue_poses):
            blue_at_goal += 1
            if blue_at_goal >= task_level:
                itr = 0
                bfs_path = expert_policy.search_path(obs)
        if len(bfs_path) > 0:
            if itr >= len(bfs_path):
                print(obs)
            action = bfs_path[itr]
            itr += 1

            if action == 4: # stuck
                print("stuck done")
                break

        obs, reward, done, _, info = env.step(action)
        masked_obs = masking_obs(obs)
        frames.append(env.render())
        # obs = obs["unmasked"]  # Always use unmasked for the expert

        trajectory["acts"].append(action)
        trajectory["infos"].append(info)
        if full_visibility:
            trajectory["obs"].append(obs.copy())
        else:
            trajectory["obs"].append(masked_obs.copy())
        # if len(bfs_path) > 0:
        #     trajectory["acts"].append(action)
        #     trajectory["infos"].append(info)
        #     if full_visibility:
        #         trajectory["obs"].append(obs.copy())
        #     else:
        #         trajectory["obs"].append(masked_obs.copy())
    # if len(trajectory["acts"])>0:
    #     trajectories.append(
    #         Trajectory(
    #             obs=np.array(trajectory["obs"]),
    #             acts=np.array(trajectory["acts"]),
    #             infos=trajectory["infos"],
    #             terminal=True,
    #         )
    #     )
    if len(trajectory["acts"])>0 and len(bfs_path)>0:

        trajectories.append(
            Trajectory(
                obs=np.array(trajectory["obs"]),
                acts=np.array(trajectory["acts"]),
                infos=trajectory["infos"],
                terminal=True,
            )
        )
    return trajectories

def collect_augmented_trajectories3(trajectories, expert_policy, bc_policy, env, beta=0.5, full_visibility=False):
    """
    Generate trajectories for the expert using unmasked observations.

    Parameters:
        expert_policy: The expert policy.
        env: The DualObservationWrapper environment.
        n_episodes: Number of episodes to collect.

    Returns:
        Trajectories collected by the expert.
    """
    obs, _  = env.reset()
    masked_obs = masking_obs_lava(obs)
    # obs = obs["unmasked"]  # Use unmasked observation for the expert
    done = False


    trajectory = {"obs": [], "acts": [], "infos": []}
    trajectory["obs"].append(masked_obs)

    while not done:
        is_expert = random.random() > beta
        if is_expert:
            action = expert_policy.predict(obs, deterministic=True)
        else:
            if full_visibility:
                action = bc_policy.predict(obs, deterministic=True)
            else:
                action = bc_policy.predict(masked_obs, deterministic=True)
        if isinstance(action, tuple):  # Handle VecEnv API
            action, _ = action
            action = action.item()
        if action == 4: # stuck
            print("stuck done")
            break



        obs, reward, done, _, info = env.step(action)
        masked_obs = masking_obs_lava(obs)
        # obs = obs["unmasked"]  # Always use unmasked for the expert
        if is_expert:
            trajectory["acts"].append(action)
            trajectory["infos"].append(info)
            if full_visibility:
                trajectory["obs"].append(obs.copy())
            else:
                trajectory["obs"].append(masked_obs.copy())
    if len(trajectory["acts"])>0:
        trajectories.append(
            Trajectory(
                obs=np.array(trajectory["obs"]),
                acts=np.array(trajectory["acts"]),
                infos=trajectory["infos"],
                terminal=True,
            )
        )
    return trajectories

def collect_augmented_trajectories_ensemble(trajectories, expert_policy, bc_ensemble, env, beta=1, full_visibility=False):
    """
    Generate trajectories for the expert using unmasked observations.

    Parameters:
        expert_policy: The expert policy.
        env: The DualObservationWrapper environment.
        n_episodes: Number of episodes to collect.

    Returns:
        Trajectories collected by the expert.
    """
    obs, _  = env.reset()
    masked_obs = masking_obs_lava(obs)
    # obs = obs["unmasked"]  # Use unmasked observation for the expert
    done = False


    trajectory = {"obs": [], "acts": [], "infos": []}
    trajectory["obs"].append(masked_obs)
    disagreement = False
    bfs_path = []
    step = 0
    epi_step = 0
    while not done:
        probabilities = []
        if not disagreement and epi_step > 0:
            for bc in bc_ensemble:
                probabilities.append(bc.get_action_probabilities(masked_obs))
            variances = np.sum(np.var(probabilities, axis=0))
            argmax_indices = np.argmax(probabilities, axis=-1)
            different_action_count = len(np.unique(argmax_indices)) - 1
            disagreement = variances+different_action_count > beta
            # print(obs)
            # print("variance:", variances)

            if disagreement:
                bfs_path = expert_policy.search_path(obs)
                step = 0
        if disagreement:
            action = bfs_path[step]
            step += 1
        else:
            if full_visibility:
                action = bc_ensemble[0].policy.predict(obs, deterministic=True)
            else:
                action = bc_ensemble[0].policy.predict(masked_obs, deterministic=True)
        if isinstance(action, tuple):  # Handle VecEnv API
            action, _ = action
            action = action.item()

        obs, reward, done, _, info = env.step(action)
        masked_obs = masking_obs_lava(obs)
        epi_step += 1
        # obs = obs["unmasked"]  # Always use unmasked for the expert
        if disagreement:
            trajectory["acts"].append(action)
            trajectory["infos"].append(info)
            if full_visibility:
                trajectory["obs"].append(obs.copy())
            else:
                trajectory["obs"].append(masked_obs.copy())
    if len(trajectory["acts"])>0:
        trajectories.append(
            Trajectory(
                obs=np.array(trajectory["obs"]),
                acts=np.array(trajectory["acts"]),
                infos=trajectory["infos"],
                terminal=True,
            )
        )
    return trajectories

def evaluate_policy(policy, env, path, n_episodes=10, collect_failure=False, full_visibility=False):
    # student_failed_states = []
    frame = []
    success_cnt = 0
    for episode in range(n_episodes):
        frames = []
        obs, _ = env.reset()
        frames.append(env.render())
        if full_visibility:
            masked_obs = obs
        else:
            masked_obs = masking_obs(obs)
        # obs = obs["unmasked"]  # Use unmasked observation for the expert
        done = False
        states = [obs]
        while not done:
            action, _ = policy.predict(masked_obs, deterministic=True)
            action = action.item()
            obs, reward, done, _, info = env.step(action)
            masked_obs = masking_obs(obs)
            states.append(obs)
            frames.append(env.render())
        # if reward == 0: #student failed
        #     student_failed_states = student_failed_states + states
        if info["goal"]:
            success_cnt += 1
        model_save_path = os.path.join(path, f"{episode}.gif")
        # model_save_dir = os.path.dirname(model_save_path)
        # os.makedirs(model_save_dir, exist_ok=True)
        imageio.mimsave(model_save_path, frames, duration=1 / 20, loop=0)
        # writer = imageio.get_writer(f"./toy_student/dagger/testing_{episode}.mp4", fps=20)
        #
        # for im in frames:
        #     writer.append_data(im)
        #
        # writer.close()
        frame.append(frames)
    evaluation={
        "success_cnt": success_cnt,
        "fail_cnt": n_episodes - success_cnt,
        "success_rate": success_cnt/n_episodes*100,
    }
    return evaluation, frame


def compare_policy(bc, dagger, env, n_episodes=10, full_visibility=False):
    frame1 = []
    frame2 = []
    success_cnt1 = 0
    success_cnt2 = 0
    bc_wins = []
    dagger_wins = []
    for episode in range(n_episodes):
        frames1 = []
        frames2 = []
        bc_succeed = False
        dagger_succeed = False

        obs, _ = env.reset()
        env2 = copy.deepcopy(env)
        obs2 = copy.deepcopy(obs)
        frames1.append(env.render())
        if full_visibility:
            masked_obs = obs
        else:
            masked_obs = masking_obs(obs)
        # obs = obs["unmasked"]  # Use unmasked observation for the expert
        done = False
        states = [obs]
        while not done:
            action, _ = bc.predict(masked_obs, deterministic=True)
            action = action.item()
            obs, reward, done, _, info = env.step(action)
            masked_obs = masking_obs(obs)
            states.append(obs)
            frames1.append(env.render())
        # if reward == 0: #student failed
        #     student_failed_states = student_failed_states + states
        if info["goal"]:
            success_cnt1 += 1
            bc_succeed = True
        frame1.append(frames1)

        frames2.append(env2.render())
        obs = obs2
        if full_visibility:
            masked_obs = obs
        else:
            masked_obs = masking_obs(obs)
        # obs = obs["unmasked"]  # Use unmasked observation for the expert
        done = False
        states = [obs]
        while not done:
            action, _ = dagger.predict(masked_obs, deterministic=True)
            action = action.item()
            obs, reward, done, _, info = env2.step(action)
            masked_obs = masking_obs(obs)
            states.append(obs)
            frames2.append(env2.render())
        # if reward == 0: #student failed
        #     student_failed_states = student_failed_states + states
        if info["goal"]:
            success_cnt2 += 1
            dagger_succeed = True
        frame2.append(frames2)
        if bc_succeed and not dagger_succeed:
            bc_wins.append(episode)
        if dagger_succeed and not bc_succeed:
            dagger_wins.append(episode)
    evaluation={
        "BC":{
            "success_cnt": success_cnt1,
            "fail_cnt": n_episodes - success_cnt1,
            "success_rate": success_cnt1 / n_episodes * 100,
            "wins": bc_wins
        },
        "DAgger": {
            "success_cnt": success_cnt2,
            "fail_cnt": n_episodes - success_cnt2,
            "success_rate": success_cnt2 / n_episodes * 100,
            "wins": dagger_wins
        },

    }
    return evaluation, frame1, frame2


def update_trajectories(trajectory, trajectories, bc_ensemble, n_ensemble):
    for i in range(n_ensemble):
        new_traj_len = len(trajectory)
        old_traj_len = len(trajectories[i])
        if new_traj_len > old_traj_len:
            trajectories[i] = trajectories[i] + trajectory[-(new_traj_len - old_traj_len):]
        bc_ensemble[i].set_demonstrations(trajectories[i])
    return bc_ensemble, trajectories

