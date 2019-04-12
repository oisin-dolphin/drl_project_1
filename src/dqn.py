import numpy as np
from collections import deque
import torch


def dqn(
    agent,
    env,
    brain_name,
    n_episodes=1000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.99,
):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[
            brain_name
        ]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state

        # state = env.reset(train_mode=True)[brain_name]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)

            env_info = env.step(action)[
                brain_name
            ]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            # next_state, reward, done, _ = env.step(action)[brain_name]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        if i_episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f} \tEpsilon: {:.2f}".format(
                    i_episode, np.mean(scores_window), eps
                )
            )
        if np.mean(scores_window) >= 13.0:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_window)
                )
            )
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
            break
    return scores
