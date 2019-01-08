import numpy as np
import pandas as pd
import h5py

class QLearningTable:
    def __init__(self, actions, load, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.render = True
        self.load_model = load

        self.actions = actions  # a list 上下左右
        self.lr = learning_rate
        self.gamma = reward_decay 
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

        if self.load_model:
            self.q_table = self.load_weights("./save_model/2048_qlearning.h5")
            print('load q_table: ', self.q_table)


    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def save_weights(self, location):
        self.q_table.to_hdf(location, key='df', mode='w')

    def load_weights(self, location):
        return pd.read_hdf(location)


