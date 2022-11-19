from torch.utils.data import Dataset
import numpy as np
import os.path as osp
import os
from typing import List, Dict, Tuple
import re
import pickle

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def eval_traj(env_name:str, env_level:str, trajs:List[Dict[str, np.array]], idx_name:str, mode:str='normal')->list:

    states, traj_lens, returns = [], [], []
    actions, rewards = [], [] ## I add this two for normalization about actions and rewards
    for path in trajs:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        actions.append(path['actions'])
        rewards.append(path['rewards'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())

    traj_lens, returns = np.array(traj_lens), np.array(returns)
    # print(traj_lens)

    num_timesteps = sum(traj_lens) ## 1M for D4RL dataset

    print('=' * 50)
    print(f'Starting new experiment: {env_name} | {env_level} | {idx_name}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    return states, actions, rewards, returns, traj_lens, num_timesteps

def get_trajectory_CheetahWorld(env_name, env_level, root, dataset_name):
    max_ep_len = 200
    scale = 1.
    state_dim = 20
    act_dim = 6

    def get_traj_from_env_levl(env_name:str, env_name_path:str, env_level:str, trajectories:List[Dict[str, np.array]]):
        level_path = osp.join(env_name_path, env_level)
        assert osp.exists(level_path), f'The {level_path} is not exist!!!'

        ## get all the task idx in each env
        ## e.g. 2 tasks in cheetah-dir and 65 tasks in chhetah-vel
        # json_path = osp.join(level_path, f"info_{env_name}_{env_level}.json")
        # with open(json_path, 'rb') as f:
        #     json_file = json.load(f)
        #     # idxs = json_file['idxs'] 
        #     state_dim = json_file['state_dim']
        #     act_dim = json_file['act_dim']

        for buffer_name in os.listdir(level_path):
            if buffer_name.endswith('.pkl'):
                # dataset_path = osp.join(level_path, f"buffer_{env_name}_{env_level}_id{idx}.pkl")
                buffer_path = osp.join(level_path, buffer_name)
                assert osp.exists(buffer_path), f'The {buffer_path} is not exist!!!'

                idx_name = re.split('_|.pkl', buffer_name)[-2]

                with open(buffer_path, 'rb') as f:
                    ## trajectories is a list containing 1K path.
                    ## Each path is a dict containing (o,a,r,no,d) with 1K steps
                    _trajs = pickle.load(f)
                    _ = eval_traj(env_name, env_level, _trajs, idx_name)
                    trajectories.extend(_trajs) ## do not use list.append()

        return trajectories

    trajectories = []
    dataset_path = osp.join(root, dataset_name)
    if env_name == 'all': 
        for _env_name in os.listdir(dataset_path):
            env_name_path = osp.join(dataset_path, _env_name)

            if env_level == 'all': 
                for _env_level in os.listdir(env_name_path):
                    trajectories = get_traj_from_env_levl(_env_name, env_name_path, _env_level, trajectories)


            else: ## env_level = one of ['normal', 'relabel', 'cic', 'rnd', 'icmapt']
                trajectories = get_traj_from_env_levl(_env_name, env_name_path, env_level, trajectories)

    else: ## env_name in ['cheetah-dir', 'cheetah-vel']
        env_name_path = osp.join(dataset_path, env_name)

        if env_level == 'all': 
                for _env_level in os.listdir(env_name_path):
                    trajectories = get_traj_from_env_levl(env_name, env_name_path, _env_level, trajectories)

        else: ## env_level = one of ['normal', 'relabel', 'cic', 'rnd', 'icmapt']
            trajectories = get_traj_from_env_levl(env_name, env_name_path, env_level, trajectories)
    return trajectories

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
        eval_traj=None,
        normalize=True
    ):
        
        self.dataset_name = dataset_name
        self.trajs = trajs
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_len = max_len
        self.scale = scale
        self.normalize = normalize
        
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

        traj = self.trajs[idx]
        
        if self.dataset_name == 'D4RL':
            pass
        elif self.dataset_name == 'CheetahWorld-v2':
            s = traj['observations']
            a = traj['actions']
            r = traj['rewards']
            d = np.zeros(s.shape[0])
            timesteps = np.arange(s.shape[0])
            rtg = discount_cumsum(traj['rewards'], gamma=1.)[:s.shape[0] + 1].reshape(-1, 1)
            if rtg.shape[0] <= s.shape[0]:
                rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

            # padding, state + reward normalization
            ## at first glance, this is strange for padding from the left,
            ## but then I realized, that it doesn't matter from the left or the right,
            ## the key point is to be consistence with mask
            tlen = s.shape[0] ## the len of each traj
            s = np.concatenate([np.zeros((self.max_len - tlen, self.state_dim)), s], axis=0) ## make s = (1, self.max_len, Ds)
            ## why use np.ones for action??
            a = np.concatenate([np.ones((self.max_len - tlen, self.act_dim)) * -10., a], axis=0) 
            r = np.concatenate([np.zeros((self.max_len - tlen, 1)), r], axis=0)
            
            if self.normalize:
                s = (s - self.state_mean) / self.state_std
                a = (a - self.action_mean) / self.action_std 
                r = (r - self.reward_mean) / self.reward_std
                
            d = np.concatenate([np.ones((self.max_len - tlen)) * 2, d], axis=0)
            rtg = np.concatenate([np.zeros((self.max_len - tlen, 1)), rtg], axis=0) / self.scale
            timesteps = np.concatenate([np.zeros((self.max_len - tlen)), timesteps], axis=0)
            mask = np.concatenate([np.zeros((self.max_len - tlen)), np.ones((tlen))], axis=0)

        return s, a, r, d, rtg, timesteps, mask
