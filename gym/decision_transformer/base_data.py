from torch.utils.data import Dataset
import numpy as np

from typing import List, Dict, Tuple

def discount_cumsum(x, gamma):
    _discount_cumsum = np.zeros_like(x)
    _discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        _discount_cumsum[t] = x[t] + gamma * _discount_cumsum[t+1]
    return _discount_cumsum


class CustomDataset(Dataset):

    # Basic Instantiation
    def __init__(self, 
        dataset_name:str, 
        env_name:str,
        env_level:str,
        trajs:List[Dict[str, np.array]],
        state_dim:str=20,
        act_dim:str=6,
        max_len:int=200,
        scale:float=1.0,
        eval_traj=None
    ):
        
        self.dataset_name = dataset_name
        self.trajs = trajs
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_len = max_len
        self.scale = scale
        
        if eval_traj is not None:
            states, actions, rewards, returns, traj_lens, num_timesteps = eval_traj(env_name, env_level, trajs, idx_name='all', mode='normal')
            
            # used for input normalization
            states = np.concatenate(states, axis=0) ## np.array (1M, ds)
            self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
            actions = np.concatenate(actions, axis=0) ## np.array (1M, da)
            self.action_mean, self.action_std = np.mean(actions, axis=0), np.std(actions, axis=0) + 1e-6
            rewards = np.concatenate(rewards, axis=0) ## np.array (1M, 1)
            self.reward_mean, self.reward_std = np.mean(rewards, axis=0), np.std(rewards, axis=0) + 1e-6
            
        
    # Length of the Dataset
    def __len__(self):
        return len(self.trajs)
    
    # Fetch an item from the Dataset
    def __getitem__(self, idx) -> Tuple[np.array]:
        '''
        return
            s: np.array (len, ds)
            a: np.array (len, da)
            r: np.array (len, 1)
            rtg: np.array (len, 1)
            timesteps: np.array (len)
            mask: np.array (len)
        '''

        traj = self.trajs[idx]
        
        if self.dataset_name == 'D4RL':
            pass
        elif self.dataset_name == 'CheetahWorld-v2':
            s = traj['observations']
            a = traj['actions']
            r = traj['rewards']
            d = np.zeros(s.shape[0])
            timesteps = np.arange(s.shape[0])
            rtg = discount_cumsum(traj['rewards'], gamma=1.).reshape(-1, 1)
            # if rtg.shape[0] <= s.shape[0]:
            #     rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)
            # rtg = discount_cumsum(traj['rewards'], gamma=1.)[:s.shape[0] + 1].reshape(-1, 1)
            # if rtg.shape[0] <= s.shape[0]:
            #     rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

            # padding, state + reward normalization
            ## at first glance, this is strange for padding from the left,
            ## but then I realized, that it doesn't matter from the left or the right,
            ## the key point is to be consistence with mask
            tlen = s.shape[0] ## the len of each traj
            s = np.concatenate([np.zeros((self.max_len - tlen, self.state_dim)), s], axis=0) ## make s = (1, self.max_len, Ds)
            s = (s - self.state_mean) / self.state_std

            ## why use np.ones for action??
            a = np.concatenate([np.ones((self.max_len - tlen, self.act_dim)) * -10., a], axis=0) 
            a = (a - self.action_mean) / self.action_std 

            r = np.concatenate([np.zeros((self.max_len - tlen, 1)), r], axis=0)
            r = (r - self.reward_mean) / self.reward_std 

            d = np.concatenate([np.ones((self.max_len - tlen)) * 2, d], axis=0)
            rtg = np.concatenate([np.zeros((self.max_len - tlen, 1)), rtg], axis=0) / self.scale
            timesteps = np.concatenate([np.zeros((self.max_len - tlen)), timesteps], axis=0)
            mask = np.concatenate([np.zeros((self.max_len - tlen)), np.ones((tlen))], axis=0)

        return s, a, r, d, rtg, timesteps, mask
