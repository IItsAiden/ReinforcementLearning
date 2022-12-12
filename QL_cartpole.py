import gym
import numpy as np
import random
import math
import time

env = gym.make('CartPole-v1')

np.random.seed(int(time.time()))

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3) #(x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = (-0.5, 0.5)
STATE_BOUNDS[3] = (-math.radians(50), math.radians(50))

# Creating a Q-Table
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

NUM_TRAIN_EPISODES = 999

def train():
    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99
    num_train_streaks = 0

    for episode in range(NUM_TRAIN_EPISODES):
        obv, _ = env.reset()
        state_0 = state_to_bucket(obv)

        for t in range(999):
            action = select_action(state_0, explore_rate)
            obv, reward, done, truncated, _ = env.step(action)
            state = state_to_bucket(obv)

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate*(cal_reward(obv[2]) + discount_factor*(best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            if done or truncated:
               if truncated:
                   num_train_streaks += 1
               else:
                   num_train_streaks = 0
               break

        # It's considered done when it's solved over 9 times consecutively
        if num_train_streaks > 9:
            print(f'Training Finish, episode used: {episode}')
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

def select_action(state, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])
    return action

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))

def cal_reward(angle):
    return 0 if angle > 0.10472 or angle < -0.10472 else 1

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":
    print('Training ...')
    train()
