import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

class A2CAgent:
	def __init__(self, state_size, action_size, load):
		self.render = True
		self.load_model = load

		self.state_size = state_size
		self.action_size = action_size
		self.value_size = 1

		self.discount_factor = 0.99
		self.ac_lr = 0.001
		self.cr_lr = 0.005

		self.actor = self.build_actor()
		self.critic = self.build_critic()

		if self.load_model:
			self.actor.load_weights("./save_model/2048_actor.h5")
			self.critic.load_weights("./save_model/2048_critic.h5")

	def build_actor(self):
		actor = Sequential()
		actor.add(Dense(4, input_dim=self.state_size, activation="relu",
			kernel_initializer="zeros"))
		actor.add(Dense(self.action_size, activation="softmax",
			kernel_initializer="zeros"))

		actor.summary()

		actor.compile(loss="categorical_crossentropy",
			optimizer=Adam(lr=self.ac_lr))
		
		return actor

	def build_critic(self):
		critic = Sequential()
		critic.add(Dense(4, input_dim=self.state_size, activation="relu",
			kernel_initializer="zeros"))
		critic.add(Dense(self.value_size, activation="linear",
			kernel_initializer="zeros"))

		critic.summary()
		critic.compile(loss="mse", optimizer=Adam(lr=self.cr_lr))
		
		return critic

	def get_action(self, state):
		policy = self.actor.predict(state, batch_size=1).flatten()
		return np.random.choice(self.action_size, 1, p=policy)[0]

	def train_model(self, state, action, reward, next_state, done):
		target = np.zeros((1, self.value_size))
		advantages = np.zeros((1, self.action_size))
		
		value = self.critic.predict(state)[0]
		next_value = self.critic.predict(next_state)[0]

		if done:
			advantages[0][action] = reward - value
			target[0][0] = reward
		else:
			advantages[0][action] = reward + self.discount_factor * (next_value) - value
			target[0][0] = reward + self.discount_factor * next_value

		self.actor.fit(state, advantages, epochs=1, verbose=0)
		self.critic.fit(state, target, epochs=1, verbose=0)