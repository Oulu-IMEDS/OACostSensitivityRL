import numpy as np
import random
from collections import namedtuple, deque

import pandas as pd
from torch.nn.utils import clip_grad_norm_

from common.model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

TAU = 1e-3              # for soft update of target parameters default 1e-3

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, cfg, model, optimizer, state_size, action_size):
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
        self.memory = ReplayBuffer(cfg, action_size, buffer_size, self.batch_size)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.state_idx = 0
        self.eps_idx = 0
    
    def step(self, state,h, action, reward, next_state, h_next, done, mode='train'):
        # Save experience in replay memory
        self.memory.add(self.eps_idx, self.state_idx, state, h, action, reward, next_state, h_next, done)
        self.state_idx += 1

        if done or self.state_idx==self.cfg.time_interval:
            self.eps_idx += 1

    def update(self):
        # Learn every UPDATE_EVERY time steps.
        # if mode == 'train':
        #     self.t_step = (self.t_step + 1) % UPDATE_EVERY
        #     if self.t_step == 0:
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size*self.cfg.time_interval:
            if self.cfg.sample_level == 'eps':
                experiences = self.memory.sample(self.eps_idx)
            elif self.cfg.sample_level == 'state':
                experiences = self.memory.sample_states()
            else:
                raise ValueError(f'Only support sample eps for state')
            self.learn(experiences, self.cfg.gamma_discount)

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
        """Update value parameters using given batch of experience tuples."""

        states, actions, rewards, next_states, dones, weights = experiences
        avg_q_target = 0
        avg_q_exp = 0
        if self.cfg.sample_level == 'eps':
            for i in range(self.cfg.time_interval):
                with torch.no_grad():
                    values_next  = self.qnetwork_target(next_states[i])
                    Q_targets_next = torch.max(values_next.detach(), dim=-1)[0].unsqueeze(-1)
                    # Compute Q targets for current states
                    q_targets = rewards[i] + (gamma * Q_targets_next * (1 - dones[i]))
                    # q_targets = rewards[i] + (gamma * Q_targets_next * (1 - 0))

                    avg_q_target += q_targets

                # Get expected Q values from local model
                values = self.qnetwork_local(states[i])
                q_expected = values.squeeze().gather(1, actions[i].type(torch.int64))
                avg_q_exp += q_expected
            Q_expected = avg_q_exp / self.cfg.time_interval
            Q_targets = avg_q_target / self.cfg.time_interval
        elif self.cfg.sample_level == 'state':
            with torch.no_grad():
                values_next = self.qnetwork_target(next_states)
                Q_targets_next = torch.max(values_next.detach(), dim=-1)[0].unsqueeze(-1).cpu()
                # Compute Q targets for current states
                Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
                # q_targets = rewards[i] + (gamma * Q_targets_next * (1 - 0))


            # Get expected Q values from local model
            values = self.qnetwork_local(states)
            Q_expected = values.cpu().gather(1, actions.type(torch.int64))
        
        # Compute loss
        loss_func = nn.MSELoss(reduction='none')
        losses = loss_func(Q_expected, Q_targets)
        # Get weighted means
        weights = torch.ones((self.cfg.batch_size,1))
        loss = torch.mean(weights * losses)

        self.optimizer.zero_grad()
        loss.backward()

        if self.cfg.clip_gradient:
            clip_grad_norm_(self.qnetwork_local.parameters(), self.cfg.clip_value)
        else:
            pass
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


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, cfg, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.cfg = cfg
        self.device = 'cpu'
        self.action_size = action_size
        self.memory = deque()
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["eps_id", "state_id", "state", "h","action", "reward", "next_state", "h_next","done"])
        # self.seed = random.seed(seed)
    
    def add(self, eps_id, state_id, state, h, action, reward, next_state, h_next, done):
        """Add a new experience to memory."""
        e = self.experience(eps_id, state_id, state, h, action, reward, next_state, h_next, done)
        self.memory.append(e)
        self.remove_if_needed()

    def remove_if_needed(self):
        if len(self.memory) > self.cfg.buffer_size:
            while True:
                eps_id = self.memory[0].eps_id
                for i in range(len(self.memory)):
                    if self.memory[0].eps_id == eps_id:
                        self.memory.popleft()
                    else:
                        break

                if len(self.memory) <= self.cfg.buffer_size:
                    break

    
    def sample(self, eps_idx):
        """Randomly sample a batch of experiences from memory."""

        # if self.cfg.balance_data_by_weight_sample:
        #     memory_df = pd.DataFrame(data=self.memory)
        #     map_freqs = memory_df['action'].value_counts(normalize=True).to_dict()
        #     # print(map_freqs)
        #     sample_weights = [1.0 / map_freqs[e] for e in memory_df['action'].tolist()]
        #     experiences = random.choices(self.memory, weights=sample_weights, k=self.batch_size)
        # else:
        #     experiences = random.sample(self.memory, k=self.batch_size)
        # keys = experiences[0].state.keys()
        #
        # states = dict.fromkeys(keys)
        # next_states = dict.fromkeys(keys)
        # for k in keys:
        #     states[f'{k}'] = torch.from_numpy(np.vstack([e.state[f'{k}'] for e in experiences if e is not None])).float().to(device)
        #     next_states[f'{k}'] = torch.from_numpy(np.vstack([e.next_state[f'{k}'] for e in experiences if e is not None])).float().to(device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
        #     device)

        """Randomly sample a batch of experiences from memory."""
        list_eps_ids = random.sample(range(self.memory[0].eps_id,eps_idx,1), k=self.batch_size)
        # print(list_eps_ids)
        # experiences = random.sample(self.memory, k=self.batch_size)

        experiences = [t for t in self.memory if t.eps_id in list_eps_ids]
        exp = []
        for id in sorted(list_eps_ids):
            exp.extend([t for t in self.memory if t.eps_id == id])

        states = []
        h = []
        next_states = []
        h_next = []
        actions = []
        rewards = []
        dones = []
        list_keys = list(experiences[0].state.keys())
        states_keys = pd.Series(dtype='float64')
        next_states_keys = pd.Series(dtype='float64')

        for j in range(self.cfg.time_interval):
            # arr_state = np.vstack([e.state for e in experiences if e.state_id == j])
            # arr_next_state = np.vstack([e.state for e in experiences if e.state_id == j])
            for i in list_keys:
                states_keys[f'{i}'] = torch.stack([e.state[f'{i}'] for e in experiences if e.state_id == j]).to(self.device)
                next_states_keys[f'{i}'] = torch.stack([e.next_state[f'{i}'] for e in experiences if e.state_id == j]).to(self.device)
            states.append(states_keys)
            next_states.append(next_states_keys)
            # states.append(torch.from_numpy(np.vstack([e.state for e in experiences if e.state_id == j])).float().to(self.device))
            # next_states.append(torch.from_numpy(np.vstack([e.next_state for e in experiences if e.state_id == j])).float().to(self.device))
            actions.append(
                torch.from_numpy(np.vstack([e.action for e in experiences if e.state_id == j])).float().to(self.device))
            rewards.append(
                torch.from_numpy(np.vstack([e.reward for e in experiences if e.state_id == j])).float().to(self.device))
            dones.append(
                torch.from_numpy(np.vstack([e.done for e in experiences if e.state_id == j])).float().to(self.device))

        tuple = (states, actions, rewards, next_states, dones)
  
        return tuple

    def sample_states(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = []
        h = []
        next_states = []
        h_next = []
        actions = []
        rewards = []
        dones = []
        list_keys = list(experiences[0].state.keys())
        states_keys = {}
        next_states_keys = {}

        for i in list_keys:
            # states_keys[f'{i}'] = torch.tensor(states_keys[f'{i}'], device=self.cfg.device)
            states_keys[f'{i}'] = torch.from_numpy(np.concatenate([e.state[f'{i}'] for e in experiences])).float()
            next_states_keys[f'{i}'] = torch.from_numpy(np.concatenate([e.next_state[f'{i}'] for e in experiences])).float()
        states = states_keys
        next_states = next_states_keys
        # states.append(torch.from_numpy(np.vstack([e.state for e in experiences if e.state_id == j])).float().to(self.device))
        # next_states.append(torch.from_numpy(np.vstack([e.next_state for e in experiences if e.state_id == j])).float().to(self.device))
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences])).float().to(self.device)
        action_clone = (actions-1).clone()
        sample_weights = 1/(torch.bincount(actions.squeeze().int())/self.cfg.batch_size)
        sample_weights = sample_weights/sample_weights.sum() * self.cfg.n_actions
        action_clone[actions==0]=1
        action_clone[actions<0]=0
        weights = sample_weights*action_clone + sample_weights*actions

        tuple = (states, actions, rewards, next_states, dones, weights)

        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return tuple


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)