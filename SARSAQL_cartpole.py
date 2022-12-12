import gym
import numpy as np
import math

env = gym.make("CartPole-v1")

#Hyperparamters
EPISODES = 30000
DISCOUNT = 0.95
EPISODE_DISPLAY = 500
LEARNING_RATE = 0.25
EPSILON = 0.2
MIN_EXPLORE_RATE = 0.01

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/125)))

def cal_reward(angle):
    return 0 if angle > 0.10472 or angle < -0.10472 else 1

#Q-Table of size theta_state_size*theta_dot_state_size*env.action_space.n
theta_minmax = env.observation_space.high[2]
theta_dot_minmax = math.radians(50)
theta_state_size = 50
theta_dot_state_size = 50
Q_TABLE = np.random.randn(theta_state_size,theta_dot_state_size,env.action_space.n)

# For stats
ep_rewards = []
ep_rewards_table = {'ep': [], 'avg': [], 'min': [], 'max': []}

def discretised_state(state):
    #state[2] -> theta
    #state[3] -> theta_dot
    discrete_state = np.array([0,0])		#Initialised discrete array
    theta_window =  ( theta_minmax - (-theta_minmax) ) / theta_state_size
    discrete_state[0] = ( state[2] - (-theta_minmax) ) // theta_window
    discrete_state[0] = min(theta_state_size-1, max(0,discrete_state[0]))

    theta_dot_window =  ( theta_dot_minmax - (-theta_dot_minmax) )/ theta_dot_state_size
    discrete_state[1] = ( state[3] - (-theta_dot_minmax) ) // theta_dot_window
    discrete_state[1] = min(theta_dot_state_size-1, max(0,discrete_state[1]))

    return tuple(discrete_state.astype(int))

num_train_streaks = 0
e_rate = get_explore_rate(0)
for episode in range(EPISODES):
    episode_reward = 0
    done = False
    obv, _ = env.reset()

    curr_discrete_state = discretised_state(obv)
    if np.random.random() > e_rate:
        action = np.argmax(Q_TABLE[curr_discrete_state])
    else:
        action = np.random.randint(0, env.action_space.n)

    while not done:
        new_state, reward, done, truncated, _ = env.step(action)
        new_discrete_state = discretised_state(new_state)

        if np.random.random() > e_rate:
            new_action = np.argmax(Q_TABLE[new_discrete_state])
        else:
            new_action = np.random.randint(0, env.action_space.n)

        if not done:
            current_q = Q_TABLE[curr_discrete_state+(action,)]
            max_future_q = Q_TABLE[new_discrete_state+(new_action,)]
            new_q = current_q + LEARNING_RATE*(cal_reward(new_state[2])+DISCOUNT*max_future_q-current_q)
            Q_TABLE[curr_discrete_state+(action,)]=new_q

        curr_discrete_state = new_discrete_state
        action = new_action

        episode_reward += reward

        if done or truncated:
           if truncated:
               num_train_streaks += 1
               print(f'{episode} streak = {num_train_streaks}')
           else:
               num_train_streaks = 0
           break

    if num_train_streaks > 9:
        print(f'Training Finish, episode used: {episode}')
        break

    ep_rewards.append(episode_reward)
    e_rate = get_explore_rate(episode)

env.close()
