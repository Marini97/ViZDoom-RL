import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from models import ActorCnn, CriticCnn

class A2CAgent():
    def __init__(self, input_shape, action_size, seed, device, gamma, alpha, beta, update_every, 
                epsilon=1, epsilon_decay=0.9995, epsilon_min=0.05,
                load_model=False, actor_model=None, critic_model=None):
        """Initialize an Agent object.
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            gamma (float): discount factor
            alpha (float): Actor learning rate
            beta (float): Critic learning rate 
            update_every (int): how often to update the network
            load_model(bool): Load model from checkpoint
            model_savefile(string): Path to model checkpoint
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_every = update_every
        self.loss = 0
        

        # Actor-Network
        self.actor_net = ActorCnn(input_shape, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)

        # Critic-Network
        self.critic_net = CriticCnn(input_shape).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.beta)
        
        if load_model:
            try:
                print("Loading model from checkpoint")
                self.actor_net = torch.load(actor_model)
                self.critic_net = torch.load(critic_model)
            except:
                print("No model checkpoint found")
                pass

        # Memory
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.masks     = []
        self.entropies = []

        self.t_step = 0

    def step(self, state, log_prob, entropy, reward, done, next_state):

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        
        value = self.critic_net(state)
        
        # Save experience in  memory
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(torch.from_numpy(np.array([reward])).to(self.device))
        self.masks.append(torch.from_numpy(np.array([1 - done])).to(self.device))
        self.entropies.append(entropy)

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
           self.learn(next_state)
           self.reset_memory()
           
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
                
    def act(self, state):
        """Returns action, log_prob, entropy for given state as per current policy."""
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_probs = self.actor_net(state)

        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        entropy = action_probs.entropy().mean()
        
        if np.random.uniform() < self.epsilon:
            action = random.choice(range(self.action_size))
            return action, log_prob, entropy

        return action.item(), log_prob, entropy

        
        
    def learn(self, next_state):
        next_state = torch.from_numpy(next_state).unsqueeze(0).to(self.device)
        next_value = self.critic_net(next_state)

        returns = self.compute_returns(next_value, self.gamma)

        log_probs = torch.cat(self.log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(self.values)

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * sum(self.entropies)
        self.loss = loss
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
    
        


    def reset_memory(self):
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.masks[:]
        del self.entropies[:]

    def compute_returns(self, next_value, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
        return returns