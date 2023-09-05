import keras.backend as k
import numpy as np
from keras.layers import Input, Dense, GRUCell
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import environment


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def main_loop(num_episodes=1000):
    """
    Eligibility traces: custom optimizer...let's use Adam instead
    Vérifier : choix de l'action sur les actions possibles
    """
    agent_game = Agent()
    agent_place = Agent(num_inputs=25, num_actions=625)
    for i in range(num_episodes):
        final = False
        placed = False
        initial = True

        agent_place.reset_memory()
        agent_game.reset_memory()
        state_place_0, state_place_1 = environment.reset()

        while not placed:
            # placed doit être False lorsque l'avant dernière pièce est placée
            action_place_0 = agent_place.get_action(state_place_0, agent_place.memory_0)
            next_state_place_0, reward_place_0, placed = environment.step_place(state_place_0, action_place_0)
            agent_place.memory_0 = agent_place.learn(state_place_0, agent_place.memory_0, action_place_0,
                                                     reward_place_0, next_state_place_0, False)
            state_place_0 = next_state_place_0
            action_place_1 = agent_place.get_action(state_place_1, agent_place.memory_1)
            next_state_place_1, reward_place_1, placed = environment.step_place(state_place_1, action_place_1)
            agent_place.memory_1 = agent_place.learn(state_place_1, agent_place.memory_1, action_place_1,
                                                     reward_place_1, next_state_place_1, False)
            state_place_1 = next_state_place_1

        action_place_0 = agent_place.get_action(state_place_0, agent_place.memory_0)
        action_place_1 = agent_place.get_action(state_place_1, agent_place.memory_1)
        state_0 = environment.get_initial_state()

        while not final:
            action_0 = agent_game.get_action(state_0, agent_game.memory_0)
            if not initial:
                next_state_1, reward_1, final = environment.step_game(state_0, action_0)
                agent_game.memory_1 = agent_game.learn(state_1, agent_game.memory_1, action_1,
                                                       reward_1, next_state_1, final)
                state_1 = next_state_1
                # Quand la partie se finit, il faut encore faire un pas pour informer l'autre joueur
            else:
                initial = False
                state_1, _, _ = environment.step_game(state_0, action_0)
            action_1 = agent_game.get_action(state_1, agent_game.memory_1)
            next_state_0, reward_0, final = environment.step_game(state_1, action_1)
            agent_game.memory_0 = agent_game.learn(state_0, agent_game.memory_0, action_0,
                                                   reward_0, next_state_0, final)
            state_0 = next_state_0

        agent_place.learn(state_place_0, action_place_0, reward_0, next_state_place_0, True)
        agent_place.learn(state_place_1, action_place_1, reward_1, next_state_place_1, True)

    agent_place.actor.save("place_actor")
    agent_place.critic.save("place_critic")
    agent_place.policy.save("place_policy")
    agent_place.actor.save("place_actor")
    agent_place.critic.save("place_critic")
    agent_place.policy.save("place_policy")


class Agent:
    """
    Nombre de positions : 60
    Nombre de pièces déplaçables : 21
    """

    def __init__(self,
                 num_inputs=60,
                 layer1=1024,
                 layer2=512,
                 num_actions=1260,
                 learning_actor=1e-5,
                 learning_critic=5e-5,
                 gamma=1):
        self.num_inputs = num_inputs
        self.layer1 = layer1
        self.layer2 = layer2
        self.num_actions = num_actions
        self.learning_actor = learning_actor
        self.learning_critic = learning_critic
        self.gamma = gamma

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(self.num_actions)]
        self.memory_0 = np.zeros((1, self.layer1))
        self.memory_1 = np.zeros((1, self.layer1))

    def reset_memory(self):
        self.memory_0 = np.zeros((1, self.layer1))
        self.memory_1 = np.zeros((1, self.layer1))

    def build_actor_critic_network(self):
        input = Input(shape=(self.num_inputs,))
        memory = Input(shape=(self.layer1,))
        delta = Input(shape=[1])
        gru, hidden = GRUCell(self.layer1)(input, memory)
        dense = Dense(self.layer2, activation='relu')(gru)
        probs = Dense(self.num_actions, activation='softmax')(dense)
        values = Dense(1, activaion='linear')(dense)

        def custom_loss(y_true, y_pred):
            out = k.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*k.log(out)

            return k.sum(-log_lik*delta)

        actor = Model(input=[input, memory, delta], output=[probs, hidden])

        actor.compile(optimizer=Adam(lr=self.learning_actor), loss=custom_loss)

        critic = Model(input=[input, memory], output=[values, hidden])

        critic.compile(optimizer=Adam(lr=self.learning_critic), loss='mean_squared_error')

        policy = Model(input=[input, memory], output=[probs, hidden])

        return actor, critic, policy

    def get_action(self, state, memory):
        state = state[np.newaxis, :]
        probabilities, _ = self.policy.predict(state, memory)
        action = np.random.choice(self.action_space, p=probabilities[0])
        return action

    def learn(self, state, memory, action, reward, next_state, final):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        critic_value, new_memory = self.critic.predict(next_state, memory)
        next_critic_value, _ = self.critic.predict(state, new_memory)

        target = reward + self.gamma*next_critic_value*(1-int(final))
        delta = target - critic_value

        actions = np.zeros((1, self.num_actions))
        actions[:, action] = 1.0

        self.actor.fit([state, memory, delta], actions)  # verbose=0
        self.critic.fit([state, memory], target)  # verbose=0

        return new_memory
