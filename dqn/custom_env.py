import gym
from gym import spaces
import random
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from data.preprocess_data import preprocess, beam_data
from common.oai_utils import SIDE_FEATURES


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""

  def __init__(self):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.current_value = 1
  def _reset(self):
        """Resets tracking metrics."""
        self.idx = -1
        self._cum_regret = 0
        self._cum_reward = 0
        self._reward_hist = []
        self._regret_hist = []
        self._cum_regret_hist = []
        self._cum_reward_hist = []

  def step(self, action):
    # Execute one time step within the environment
    self.idx += 1
    if not self.idx < self.len:
        self.idx = 0

  def reset(self):
    # Reset the state of the environment to an initial state
    raise NotImplementedError

class CostOASensitivityEnv(CustomEnv):
  """Custom Environment that follows gym interface"""

  def __init__(self, cfg, df, n_actions, n_steps, mode='train',test_site='E'):
    super(CostOASensitivityEnv, self).__init__()
    self.cfg = cfg
    self.n_steps = n_steps
    self.n_actions = n_actions
    self.encode_visit = ['0', '12', '24', '36', '48', '72', '96']
    self.mode = mode
    self.test_site = test_site

    self.df = self.preprocess(df, mode)
    self.len = len(self.df)
    self.n_features = len(self.prefix_header) + len(self.cfg.fix_features) + 2

    self.remaining_steps = self.n_steps
    self.idx = -1
    self.eps_idx = -1
    if self.cfg.data_level == 'patient':
        self.reward_function = self._compute_reward
    elif self.cfg.data_level == 'knee':
        self.reward_function = self._compute_reward_knee_level

    # Define action and observation space
    self.observation_space = spaces.Box(-np.inf,np.inf,shape=(self.n_features, ), dtype=np.float32)
    self.action_space = spaces.Discrete(self.n_actions, start=1)
    print(self.observation_space.shape)

  def preprocess(self, df, mode):
      self.unchanged_features_header = self.cfg.fix_features
      side_features = set(SIDE_FEATURES) & set(self.cfg.set_features) - set(self.cfg.reward_feature)
      if self.cfg.data_level == 'patient':
        list_side_features = [x + '_0' for x in side_features] + [x + '_1' for x in side_features]
      elif self.cfg.data_level == 'knee':
        list_side_features = list(side_features)
      list_patient_features = list(set(self.cfg.set_features.copy()) - set(SIDE_FEATURES))

      self.prefix_header = sorted(list_patient_features + list_side_features)
      print(self.prefix_header)
      if mode == 'train':
          df = df[df['Site']!=self.cfg.test_site]
      elif mode == 'test':
          df = df[df['Site']==self.cfg.test_site]
      elif mode == 'test-action':
          df = df
      else:
          raise ValueError(f'We decide eliminate mode validation, only support train/test')

      return df

  def reset(self, mode='train'):
    # Reset the state of the environment to an initial state
    #  self._reset()
     self.idx += 1
     self.eps_idx += 1

     if not self.idx < self.len:
            self.idx = 0
            self.df = self.df.sample(frac=1).reset_index(drop=True)
      # Set the current step to a random point within the data frame
     self.info_step = 0
     self.current_step = 0
     self.latest_visit = 0
     self.remaining_steps = self.n_steps
     self.flag_stop_game = False

     self.data = self.df.iloc[self.idx]
     if self.cfg.multi_progression:
         self.len_t_p = len(self.data['Progress'])
         self.update_tp_times = self.len_t_p - 1
         if self.len_t_p  == 0:
             self.t_p = self.n_steps + 1
             self.pass_progression = True
         elif self.len_t_p > 0:
             self.t_p = self.data['Progress'][0]
             self.pass_progression = False
     else:
         self.t_p = self.data['Progress'].astype(int)
         self.len_t_p = 0
         self.update_tp_times = 0

     obs = self._next_observation()
     return obs

  def step(self, action):
    # Execute one time step within the environment
    reward, qol, self.ref_tp = self.reward_function(action)
    done = self.check_finish_condition(action)

    self.current_step += 1
    if self.cfg.learning_data == 'dynamic':
        if action == 1:
            self.info_step = self.current_step
            self.latest_visit = self.info_step
            state = self._next_observation()
        else:
            state = self._next_observation()
    elif self.cfg.learning_data == 'static':
        if action == 1:
            self.latest_visit = self.current_step
        state = self._next_observation()

    self.remaining_steps -= 1

    # self.current_step = action
    return state, reward, done, qol

  def step_for_test(self, action):
    # Execute one time step within the environment
    reward, qol, self.ref_tp = self._compute_reward_for_test(action)
    done = self.check_finish_condition(action)

    self.current_step += 1
    if action == 1:
        self.info_step = self.current_step
        state = self._next_observation()
    else:
        state = self._next_observation()

    self.remaining_steps -= 1

    return state, reward, done, qol

  def _next_observation(self):
    # Get the data points for the last 5 days and scale to between 0-1
      postfix_header = f'@{self.encode_visit[self.info_step]}'
      step_features = [x + postfix_header for x in self.prefix_header]
      step_features.extend(self.unchanged_features_header)

      obs = self.data[step_features]
      obs.rename({key: key.split(f'@{self.info_step}')[0] for key in obs.keys()}, inplace=True)
      if self.cfg.progressor_type in ['num','noImage']:
          for i in list(obs.keys()):
              if 'KL' in i:
                  obs[f'{i}'] = self.categorize_KL(obs[f'{i}'])
              elif 'IMG' in i:
                  obs[f'{i}'] = self.transform_img(obs[i])
              else:
                obs[f'{i}'] = obs[f'{i}'][None, None]
      elif self.cfg.progressor_type in ['prob','logit','feature']:
          for i in list(obs.keys()):
              if 'KL' in i:
                  obs[f'{i}'] = obs[f'{i}'][None,:]
              else:
                  obs[f'{i}'] = obs[f'{i}'][None, None]
      else:
          raise ValueError(f'Not support this KL type')
      obs['Info_step'] = np.array(self.info_step)[None, None]
      obs['Current_step'] = np.array(self.current_step)[None, None]

      return obs

  def categorize_KL(self, KL_num):
      n_segments = 5
      if KL_num == -1:
          kl = np.ones((1, n_segments))*-1
      else:
          kl_level = KL_num.astype(int)
          kl = np.zeros((1, n_segments))
          kl[:, kl_level] = 1.0
      return kl

  def _collect_target(self):
      target = self.t_p
      return target

  def check_finish_condition(self, action):


      if self.remaining_steps == 1 or self.flag_stop_game:
          done = True
      else:
          done = False

      return done


  def _compute_reward(self, action, mode='train'):
    t_p = int(self.t_p)
    t = self.current_step + 1

    self.cfg.use_weight_gamma = False
    weight_fl = 1
    if self.cfg.use_exp:
        if self.cfg.use_weight_gamma:
            weight_var = self.cfg.weight_feature[0]
            weight_gamma = torch.sigmoid(torch.tensor(self.data[f'{weight_var}@{self.encode_visit[t]}'] - self.data[
                f'{weight_var}@{self.encode_visit[self.current_step]}'])).detach().cpu().numpy()
        else:
            weight_gamma = 1
    else:
        weight_gamma = 0

    scale_t = weight_gamma*np.abs((t-t_p)/self.n_steps)
    # scale_t = 0
    phi_var = self.cfg.reward_feature[0]
    phi_t_r_side0 = self.data[f'{phi_var}_0@{self.encode_visit[self.latest_visit]}']
    phi_t_r_side1 = self.data[f'{phi_var}_1@{self.encode_visit[self.latest_visit]}']
    phi_t_side_0 = self.data[f'{phi_var}_0@{self.encode_visit[t]}']
    phi_t_side_1 = self.data[f'{phi_var}_1@{self.encode_visit[t]}']

    if action == 0:
        if t_p > self.n_steps:
            name = 'true_dismiss'
            r = self.cfg.cost.true_dm_coef*self.cfg.cost.cost_convert_r
            medical_reward = 0
        else:
            phi_t_p_side0 = self.data[f'{phi_var}_0@{self.encode_visit[t_p]}']
            phi_t_p_side1 = self.data[f'{phi_var}_1@{self.encode_visit[t_p]}']
            if t < t_p:
                name = 'true_dismiss'
            #   r = (self.cfg.cost.JSW_cost_th)*self.cfg.cost.cost_convert_r
                r = self.cfg.cost.true_dm_coef*self.cfg.cost.cost_convert_r
                medical_reward = 0
            elif t >= t_p:
                eval = 'false_dismiss'
                delta_B_side0 = max(0, phi_t_r_side0 - phi_t_p_side0)
                delta_B_side1 = max(0, phi_t_r_side1 - phi_t_p_side1)
                if self.cfg.distance_knee == 'both':
                    delta_B = np.sqrt(np.square(delta_B_side0) + np.square(delta_B_side1))
                elif self.cfg.distance_knee == 'one':
                    delta_B = max(delta_B_side0, delta_B_side1)
                r = -np.exp(scale_t) * (delta_B)*self.cfg.cost.cost_convert_r
                if self.cfg.early_stop:
                    self.flag_stop_game = True
                medical_reward = 0
    elif action == 1:
        if t_p > self.n_steps:
            eval = 'early_follow'
            delta_A_side0 = max(0, phi_t_r_side0 - phi_t_side_0)
            delta_A_side1 = max(0, phi_t_r_side1 - phi_t_side_1)
            delta_A = max(delta_A_side0, delta_A_side1)
            if delta_A >= self.cfg.cost.JSW_cost_th:
                r = np.exp(-scale_t) * delta_A
                medical_reward = delta_A
            else:
                r = 0
                medical_reward = 0
            r = self.cfg.cost.cost_convert_r * r - self.cfg.cost.hospital_cost
            self.latest_delta = self.cfg.cost.JSW_cost_th
        else:
            phi_t_p_side0 = self.data[f'{phi_var}_0@{self.encode_visit[t_p]}']
            phi_t_p_side1 = self.data[f'{phi_var}_1@{self.encode_visit[t_p]}']
            if t < t_p:
                eval = 'early_follow'
                delta_A_side0 = max(0, phi_t_r_side0 - phi_t_side_0)
                delta_A_side1 = max(0, phi_t_r_side1 - phi_t_side_1)
                delta_A = max(delta_A_side0, delta_A_side1)
                if delta_A >= self.cfg.cost.JSW_cost_th:
                    r = np.exp(-scale_t) * delta_A
                    medical_reward = delta_A
                else:
                    r = 0
                    medical_reward = 0
                r = self.cfg.cost.cost_convert_r * r - self.cfg.cost.hospital_cost
                self.latest_delta = self.cfg.cost.JSW_cost_th

            elif t == t_p:
                eval = 'true_follow'
                delta_A_side0 = max(0, phi_t_r_side0 - phi_t_side_0)
                delta_A_side1 = max(0, phi_t_r_side1 - phi_t_side_1)
                delta_A = max(delta_A_side0, delta_A_side1)
                if delta_A >= self.cfg.cost.JSW_cost_th:
                    delta_A = delta_A
                    medical_reward = delta_A
                else:
                    delta_A = 0
                    medical_reward = 0
                r = self.cfg.cost.cost_convert_r * delta_A - self.cfg.cost.hospital_cost
                medical_reward = delta_A
                self.latest_delta = delta_A

                if self.len_t_p > 1 and self.update_tp_times != 0:
                    self.t_p = self.data['Progress'][self.len_t_p - self.update_tp_times]
                    self.update_tp_times -= 1
                    self.pass_progression = False
                else:
                    self.t_p = self.n_steps + 1

            elif t > t_p:
                eval = 'late_follow'
                delta_B_side0 = max(0, phi_t_r_side0 - phi_t_p_side0)
                delta_B_side1 = max(0, phi_t_r_side1 - phi_t_p_side1)
                delta_B = max(delta_B_side0, delta_B_side1)
                r = -self.cfg.cost.late_fl_coef * np.exp(scale_t) * delta_B
                r = self.cfg.cost.cost_convert_r * r - self.cfg.cost.hospital_cost
                medical_reward = 0

                self.latest_delta = delta_B

                if self.cfg.early_stop:
                    self.flag_stop_game = True

                if self.len_t_p > 1 and self.update_tp_times != 0:
                    self.t_p = self.data['Progress'][self.len_t_p - self.update_tp_times]
                    self.update_tp_times -= 1
                else:
                    self.t_p = self.n_steps + 1

    reward = r

    return reward, medical_reward, t_p

  def _compute_reward_knee_level(self, action, mode='train'):
      t_p = int(self.t_p)
      t = self.current_step + 1
      if  self.cfg.cost_func_ver == 1:
          self.cfg.use_weight_gamma = False
          if self.cfg.use_exp:
              if self.cfg.use_weight_gamma:
                  weight_var = self.cfg.weight_feature[0]
                  weight_gamma = torch.sigmoid(torch.tensor(self.data[f'{weight_var}@{self.encode_visit[t]}'] - self.data[
                      f'{weight_var}@{self.encode_visit[self.current_step]}'])).detach().cpu().numpy()
              else:
                  weight_gamma = 1
          else:
              weight_gamma = 0

          scale_t = weight_gamma*np.abs((t-t_p)/self.n_steps)
          # scale_t = 0
          phi_var = self.cfg.reward_feature[0]
          phi_t_r = self.data[f'{phi_var}@{self.encode_visit[self.info_step]}']
          phi_t = self.data[f'{phi_var}@{self.encode_visit[t]}']

          if action == 0:
              if t_p > self.n_steps:
                  name = 'true_dismiss'
                  r = self.cfg.cost.true_dm_coef*self.cfg.cost.hospital_cost
                  medical_reward = 0
              else:
                  phi_t_p = self.data[f'{phi_var}@{self.encode_visit[t_p]}']
                  if t < t_p:
                      name = 'true_dismiss'
                      r = self.cfg.cost.true_dm_coef*self.cfg.cost.hospital_cost
                      medical_reward = 0
                  elif t >= t_p:
                      eval = 'false_dismiss'
                      delta_B = max(0, phi_t_r - phi_t_p)
                      r = -np.exp(scale_t) * (delta_B)*self.cfg.cost.cost_convert_r
                      if self.cfg.early_stop:
                          self.flag_stop_game = True
                      medical_reward = 0
          elif action == 1:
              if t_p > self.n_steps:
                  eval = 'early_follow'
                  delta_A = max(0, phi_t_r- phi_t)
                  if delta_A >= self.cfg.cost.JSW_cost_th:
                      r = np.exp(-scale_t) * delta_A
                      medical_reward = delta_A
                  else:
                      r = 0
                      medical_reward = 0
              else:
                  phi_t_p = self.data[f'{phi_var}@{self.encode_visit[t_p]}']
                  if t < t_p:
                      eval = 'early_follow'
                      delta_A = max(0, phi_t_r - phi_t)
                      if delta_A >= self.cfg.cost.JSW_cost_th:
                          r = np.exp(-scale_t) * delta_A
                          medical_reward = delta_A
                      else:
                          r = 0
                          medical_reward = 0

                  elif t == t_p:
                      eval = 'true_follow'
                      delta_A = max(0, phi_t_r - phi_t)
                      r = np.exp(-scale_t) * delta_A
                      medical_reward = delta_A

                      if self.len_t_p > 1 and self.update_tp_times != 0:
                          self.t_p = self.data['Progress'][self.len_t_p - self.update_tp_times]
                          self.update_tp_times -= 1
                          self.pass_progression = False
                      else:
                          self.t_p = self.n_steps + 1

                  elif t > t_p:
                      eval = 'late_follow'
                      delta_B = max(0, phi_t_r - phi_t_p)
                      r = -self.cfg.cost.late_fl_coef * np.exp(scale_t) * delta_B
                      medical_reward = 0

                      if self.cfg.early_stop:
                          self.flag_stop_game = True

                      if self.len_t_p > 1 and self.update_tp_times != 0:
                          self.t_p = self.data['Progress'][self.len_t_p - self.update_tp_times]
                          self.update_tp_times -= 1
                      else:
                          self.t_p = self.n_steps + 1


          if action == 0:
              reward = r
              # qol_sf12 = coef/100 * self.cfg.sf12_interval
          elif action == 1:
              reward = self.cfg.cost.cost_convert_r * r - self.cfg.cost.hospital_cost
      elif self.cfg.cost_func_ver == 3:
          weight_gamma = 1

          scale_t = np.abs((t - t_p) / self.n_steps)
          # scale_t = 0
          phi_var = self.cfg.reward_feature[0]
          phi_t_r = self.data[f'{phi_var}@{self.encode_visit[self.info_step]}']
          phi_t = self.data[f'{phi_var}@{self.encode_visit[t]}']

          if action == 0:
              if t_p > self.n_steps:
                  name = 'true_dismiss'
                  r =  self.cfg.cost.true_dm_coef * self.cfg.cost.cost_convert_r
                  medical_reward = 0
              else:
                  phi_t_p = self.data[f'{phi_var}@{self.encode_visit[t_p]}']
                  if t < t_p:
                      name = 'true_dismiss'
                      r = self.cfg.cost.true_dm_coef * self.cfg.cost.hospital_cost
                      medical_reward = 0
                  elif t >= t_p:
                      eval = 'false_dismiss'
                      delta_B = max(0, phi_t_r - phi_t_p)
                      r = -np.exp(scale_t) * (delta_B) * self.cfg.cost.cost_convert_r
                      if self.cfg.early_stop:
                          self.flag_stop_game = True
                      medical_reward = 0
          elif action == 1:
              if t_p > self.n_steps:
                  eval = 'early_follow'
                  delta_A = max(0, phi_t_r - phi_t)
                  if delta_A >= self.cfg.cost.JSW_cost_th:
                      r = np.exp(-scale_t) * delta_A
                      medical_reward = delta_A
                  else:
                      r = 0
                      medical_reward = 0
              else:
                  phi_t_p = self.data[f'{phi_var}@{self.encode_visit[t_p]}']
                  if t < t_p:
                      eval = 'early_follow'
                      delta_A = max(0, phi_t_r - phi_t)
                      if delta_A >= self.cfg.cost.JSW_cost_th:
                          r = np.exp(-scale_t) * delta_A
                          medical_reward = delta_A
                      else:
                          r = 0
                          medical_reward = 0

                  elif t == t_p:
                      eval = 'true_follow'
                      delta_A = max(0, phi_t_r - phi_t)
                      r = np.exp(-scale_t) * delta_A
                      medical_reward = delta_A

                      if self.len_t_p > 1 and self.update_tp_times != 0:
                          self.t_p = self.data['Progress'][self.len_t_p - self.update_tp_times]
                          self.update_tp_times -= 1
                          self.pass_progression = False
                      else:
                          self.t_p = self.n_steps + 1

                  elif t > t_p:
                      eval = 'late_follow'
                      delta_B = max(0, phi_t_r - phi_t_p)
                      r = -self.cfg.cost.late_fl_coef * np.exp(scale_t) * delta_B
                      medical_reward = 0

                      if self.cfg.early_stop:
                          self.flag_stop_game = True

                      if self.len_t_p > 1 and self.update_tp_times != 0:
                          self.t_p = self.data['Progress'][self.len_t_p - self.update_tp_times]
                          self.update_tp_times -= 1
                      else:
                          self.t_p = self.n_steps + 1

          if action == 0:
              reward = r
              # qol_sf12 = coef/100 * self.cfg.sf12_interval
          elif action == 1:
              reward = self.cfg.cost.cost_convert_r * r - weight_gamma * self.cfg.cost.hospital_cost


      return reward, medical_reward, t_p

  def _compute_reward_for_test(self, action):
      t_p = int(self.t_p)
      t = self.current_step + 1

      scale_t = np.abs((t - t_p) / self.n_steps)
      qualy = 0.02

      gained = 0
      loss = 0


      if action == 0:
          if t_p > self.n_steps:
              name = 'true_dismiss'
              r = self.cfg.cost.true_dm_coef * self.cfg.cost.hospital_cost
              medical_reward = 0
          else:
              if t < t_p:
                  name = 'true_dismiss'
                  r = self.cfg.cost.true_dm_coef * self.cfg.cost.hospital_cost
                  medical_reward = 0
              elif t >= t_p:
                  eval = 'false_dismiss'
                  r = -np.exp(scale_t) * (qualy) * self.cfg.cost.cost_convert_r
                  medical_reward = 0
      elif action == 1:
          if t_p > self.n_steps:
              eval = 'early_follow'
              r = - self.cfg.cost.hospital_cost
              medical_reward = 0
          else:
              if t < t_p:
                  eval = 'early_follow'
                  r = - self.cfg.cost.hospital_cost
                  medical_reward = 0

              elif t == t_p:
                  eval = 'true_follow'
                  r = np.exp(-scale_t) * qualy * self.cfg.cost.cost_convert_r
                  medical_reward = qualy

                  if self.len_t_p > 1 and self.update_tp_times != 0:
                      self.t_p = self.data['Progress'][self.len_t_p - self.update_tp_times]
                      self.update_tp_times -= 1
                      self.pass_progression = False
                  else:
                      self.t_p = self.n_steps + 1

              elif t > t_p:
                  eval = 'late_follow'
                  r = self.cfg.cost.late_fl_coef * np.exp(-scale_t) * qualy * self.cfg.cost.cost_convert_r
                  medical_reward = self.cfg.cost.late_fl_coef * qualy


                  if self.len_t_p > 1 and self.update_tp_times != 0:
                      self.t_p = self.data['Progress'][self.len_t_p - self.update_tp_times]
                      self.update_tp_times -= 1
                  else:
                      self.t_p = self.n_steps + 1

      if action == 0:
          reward = r
          # qol_sf12 = coef/100 * self.cfg.sf12_interval
      elif action == 1:
          reward = r

      return reward, medical_reward, t_p