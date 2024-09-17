import numpy as np
import random
from collections import namedtuple, deque

import pandas as pd
from torch.nn.utils import clip_grad_norm_

from common.model import QNetwork, QNetwork_KL_prob
from dqn.buffer import PrioritizedReplayBuffer
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e3)  # replay buffer size default 1e5
BATCH_SIZE = 64      # minibatch size
GAMMA = 0.9          # discount factor
TAU = 1e-3              # for soft update of target parameters default 1e-3
LR = 5e-4               # learning rate
UPDATE_EVERY = 100        # how often to update the network

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, cfg, model, optimizer, state_size, action_size):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.cfg = cfg
        self.device = self.cfg.device
        self.state_size = state_size
        self.action_size = action_size
        # self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = model
        self.qnetwork_target = model

        self.optimizer = optimizer

        # Replay memory
        self.batch_size = int(cfg.batch_size)
        buffer_size = int(cfg.buffer_size)
        self.replay_buffer = PrioritizedReplayBuffer(cfg, 2**14, self.cfg.PRB.alpha)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.state_idx = 0
        self.eps_idx = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.replay_buffer.add(state,action, reward, next_state, done)
        self.state_idx += 1

        if done or self.state_idx==self.cfg.time_interval:
            self.eps_idx += 1

    def update(self):
        # Learn every UPDATE_EVERY time steps.
        # If enough samples are available in memory, get random subset and learn
        if self.replay_buffer.is_full():
                # # Train the model
                # self.train(beta)
            experiences = self.replay_buffer.sample(self.cfg.batch_size, self.cfg.PRB.beta)
            self.learn(experiences, GAMMA)

    def act(self, arr_state, mode, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if self.state_idx == 0:
            self.h = torch.zeros((2*self.cfg.RNN.n_layers,self.cfg.RNN.hidden_size)).to(self.cfg.device)
            self.prev_action = None
        self.qnetwork_local.eval()
        #Convert to Tensor
        state = {}
        for i in list(arr_state.keys()):
            state[f'{i}'] = torch.tensor(arr_state[f'{i}'], device=self.cfg.device).float()

        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        action_probs = F.softmax(action_values, dim=-1)

        if mode == 'train':
            # Epsilon-greedy action selection
            if random.random() > eps:
                action = np.argmax(action_values.cpu().data.numpy())
            else:
                action = random.choice(np.arange(self.action_size))
        elif mode == 'val' or mode == 'test':
            action = np.argmax(action_values.cpu().data.numpy())
        elif mode == 'follow':
            action = 1
        elif mode == 'dismiss':
            action = 0
        elif mode == '2year':
            if self.state_idx % 2 == 0:
                action = 1
            else:
                action = 0
        elif mode == 'random':
            action = random.choice(np.arange(self.action_size))
        elif mode == 'hypo-guidelines':
            if max(np.argmax(state['KL_0'].detach().numpy()),np.argmax(state['KL_1'].detach().numpy())) >= 3:
                action = 1
            else:
                if self.state_idx % 2 == 0:
                    action = 1
                else:
                    action = 0
        else:
            raise ValueError(f'Only support mode train/val/test/follow/dismiss')
        self.prev_action = action

        # if action != 0:
        #     action = action + self.state_idx


        return action, action_probs

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, sample_indexes = experiences
        avg_q_target = 0
        avg_q_exp = 0

        # Get expected Q values from local model
        values = self.qnetwork_local(states)
        Q_expected = values.cpu().gather(1, actions.type(torch.int64))

        with torch.no_grad():
            values_next = self.qnetwork_target(next_states)
            Q_targets_next = torch.max(values_next.detach(), dim=-1)[0].unsqueeze(-1).cpu()
            # Compute Q targets for current states
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            #Compute TD error
            td_error = Q_expected - Q_targets

        # Compute loss
        loss_func = nn.SmoothL1Loss(reduction='none')
        losses = loss_func(Q_expected, Q_targets)
        # Get weighted means
        loss = torch.mean(weights * losses)
        # loss = F.mse_loss(Q_expected, Q_targets)

        # Calculate priorities for replay buffer $p_i = |\delta_i| + \epsilon$
        new_priorities = np.abs(td_error.cpu().numpy()) + 1e-6
        # Update replay buffer priorities
        self.replay_buffer.update_priorities(sample_indexes, new_priorities)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(self.qnetwork_local.parameters(), self.cfg.clip_value)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
