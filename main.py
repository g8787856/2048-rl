from argparse import ArgumentParser
from env import Game
from Qlearning import QLearningTable
from a2c import A2CAgent
import matplotlib
matplotlib.use("TkAgg")
import pylab
import numpy as np
import sys

parser = ArgumentParser()
parser.add_argument("-algorithm",  dest="rl_algorithm", default="Qlearning")
parser.add_argument("-load", dest="load_modal", default=True)

args = parser.parse_args()

EPISODES = 1000

env = Game()
state_size = env.n_features
action_size = env.n_actions

if __name__ == "__main__":
    print('rl_algorithm: {}'.format(args.rl_algorithm))
    print('load_modal: {}'.format(args.load_modal))
    if args.rl_algorithm == 'a2c':
        agent = A2CAgent(state_size, action_size, load=args.load_modal)

        scores, episodes = [], []

        for e in range(EPISODES):
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])

            while not done:
                if agent.render:
                    env.render()

                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                
                next_state = np.reshape(next_state, [1, state_size])

                reward = reward if not done or score == 499 else -100

                agent.train_model(state, action, reward, next_state, done)

                score += reward
                state = next_state

                if done:
                    score = score if score == 500.0 else score + 100
                    scores.append(score)
                    episodes.append(e)
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("./save_graph/2048_a2c.png")
                    print("episode: ", e, " socre:", score)

            if e % 50 == 0:
                agent.actor.save_weights("./save_model/2048_actor.h5")
                agent.critic.save_weights("./save_model/2048_critic.h5")

    elif args.rl_algorithm == 'Qlearning':
        RL = QLearningTable(actions=list(range(env.n_actions)), load=args.load_modal)

        scores, episodes = [], []
        
        for e in range(EPISODES):
            done = False
            score = 0
            state = env.reset()

            while not done:
                if RL.render:
                    env.render()

                action = RL.choose_action(str(state))
                next_state, reward, done = env.step(action)

                reward = reward if not done or score == 499 else -100

                RL.learn(str(state), action, reward, str(next_state))

                score += reward
                state = next_state

                if done:
                    score = score if score == 500.0 else score + 100
                    scores.append(score)
                    episodes.append(e)
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("./save_graph/2048_qlearning.png")
                    print("episode: ", e, " socre:", score)
                    
            if e % 50 == 0:
                RL.save_weights("./save_model/2048_qlearning.h5")

    # end of game
    print('game over')
    env.destroy()