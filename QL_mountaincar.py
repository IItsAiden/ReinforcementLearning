import gym
import numpy as np
import math

env = gym.make("MountainCar-v0")

DISCRETE_BUCKETS = 20
EPISODES = 30000
DISCOUNT = 0.95
EPISODE_DISPLAY = 500
LEARNING_RATE = 0.1
EPSILON = 0.5
EPSILON_DECREMENTER = EPSILON/(EPISODES//4)
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

#Q-Table of size DISCRETE_BUCKETS*DISCRETE_BUCKETS*env.action_space.n
Q_TABLE = np.random.randn(DISCRETE_BUCKETS,DISCRETE_BUCKETS,env.action_space.n)

def cal_reward(x, t):
    if x < 0.36:
        return 0
    elif x > 0.45:
        return 45
    elif x == 0.50:
        if t < 180:
            return 150
        elif t < 150:
            return 210
        else :
            return 100
    else :
        return 1

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/75)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))

def discretised_state(state):
    DISCRETE_WIN_SIZE = (env.observation_space.high-env.observation_space.low)/[DISCRETE_BUCKETS]*len(env.observation_space.high)
    discrete_state = (state-env.observation_space.low)//DISCRETE_WIN_SIZE
    return tuple(discrete_state.astype(np.int32))		#integer tuple as we need to use it later on to extract Q table values

e_rate = get_explore_rate(0)
l_rate = get_explore_rate(0)
num_train_streaks = 0
for episode in range(EPISODES):
    episode_reward = 0
    done = False
    t = 0

    obv, _ = env.reset()
    curr_discrete_state = discretised_state(obv)

    for t in range(600):
#    while not done:
        if np.random.random() > e_rate:
            action = np.argmax(Q_TABLE[curr_discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, truncated, _ = env.step(action)
        new_discrete_state = discretised_state(new_state)

        #if not done:
        max_future_q = np.max(Q_TABLE[new_discrete_state])
        current_q = Q_TABLE[curr_discrete_state+(action,)]
        new_q = current_q + LEARNING_RATE*(cal_reward(new_state[0], t)+ DISCOUNT*max_future_q - current_q)
        Q_TABLE[curr_discrete_state+(action,)]=new_q
        #elif new_state[0] >= env.goal_position:
        #    Q_TABLE[curr_discrete_state + (action,)] = 0

        curr_discrete_state = new_discrete_state
        episode_reward += reward
        if done or truncated:
            if truncated:
#                print(f'{episode}, time = {t}')
                num_train_streaks = 0
            else:
                num_train_streaks += 1
                print(f'{episode}, time = {t}, streak = {num_train_streaks}')
            break
    if num_train_streaks > 9:
        print(f'Training Finish, episode used: {episode}')
        break

    e_rate= get_explore_rate(episode)
    l_rate = get_explore_rate(episode)




env.close()
