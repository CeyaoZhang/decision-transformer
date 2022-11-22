from torch.utils.data import Dataset
import numpy as np

import gym

import os 
import os.path as osp
import pickle
import re

from typing import List, Dict, Tuple

def discount_cumsum(x, gamma):
    _discount_cumsum = np.zeros_like(x)
    _discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        _discount_cumsum[t] = x[t] + gamma * _discount_cumsum[t+1]
    return _discount_cumsum


def eval_traj(env_name:str, env_level:str, trajs:List[Dict[str, np.array]], idx_name:str, mode:str='normal')->list:

    task_idx = idx_name.split('id')[-1]

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

        if task_idx == 'all':
            pass
        else:
            try:
                np.int32(task_idx)
            except ValueError:
                task_idx = task_idx.split('to')[1]
            path['idx'] = np.int32(task_idx)

    rewards, returns = np.array(rewards), np.array(returns)
    # print(traj_lens)
    traj_lens = np.array(traj_lens)
    num_timesteps = sum(traj_lens) ## 1M for D4RL dataset

    print('=' * 50)
    print(f'Starting new experiment: {env_name} | {env_level} | {idx_name}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print('-' * 50)
    print(f'Average reward: {np.mean(rewards):.2f}, std: {np.std(rewards):.2f}')
    print(f'Max reward: {np.max(rewards):.2f}, min: {np.min(rewards):.2f}')
    print('-' * 50)
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    return states, actions, rewards, returns, traj_lens, num_timesteps


def get_traj_from_dataset(dataset_name, env_name, env_level, model_type)->List[Dict[str, np.array]]:
    if dataset_name == 'D4RL':
        if env_name == 'hopper':
            env = gym.make('Hopper-v3')
            max_ep_len = 1000
            env_targets = [3600, 1800]  # evaluation conditioning targets
            scale = 1000.  # normalization for rewards/returns
        elif env_name == 'halfcheetah':
            env = gym.make('HalfCheetah-v3')
            max_ep_len = 1000
            env_targets = [12000, 6000]
            scale = 1000.
        elif env_name == 'walker2d':
            env = gym.make('Walker2d-v3')
            max_ep_len = 1000
            env_targets = [5000, 2500]
            scale = 1000.
        elif env_name == 'reacher2d':
            from decision_transformer.envs.reacher_2d import Reacher2dEnv
            env = Reacher2dEnv()
            max_ep_len = 100
            env_targets = [76, 40]
            scale = 10.
        else:
            raise NotImplementedError

        if model_type == 'bc':
            env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

        # load dataset
        # dataset_path = f'data/{dataset}/{env_name}-{env_level}-v2.pkl'
        dataset_path = osp.join("data", dataset_name, f"{env_name}-{env_level}-v2.pkl")
        assert osp.exists(dataset_path), 'The path is not exist!!!'
        with open(dataset_path, 'rb') as f:
            ## trajectories is a list containing 1K path.
            ## Each path is a dict containing (o,a,r,no,d) with 1K steps
            trajectories = pickle.load(f)

        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
    elif dataset_name == 'CheetahWorld-v2':

        max_ep_len = 200
        scale = 1.
        state_dim = 20
        act_dim = 6
        env_targets = None

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
                
            for buffer_name in os.listdir(level_path): ## include all task idx
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
        dataset_path = osp.join("data", dataset_name)
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
    
    print(f"state_dim: {state_dim} & act_dim: {act_dim}")

    return trajectories, (state_dim, act_dim, max_ep_len, env_targets)


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
    def __getitem__(self, idx) -> Tuple[List[np.array], np.array]:
        '''
        return
            feature: list
                s: np.array (len, ds)
                a: np.array (len, da)
                r: np.array (len, 1)
                rtg: np.array (len, 1)
                timesteps: np.array (len)
                mask: np.array (len)
            label: np.array
                task_id: np.array()
        '''

        traj = self.trajs[idx]
        
        if self.dataset_name == 'D4RL':
            pass
        elif self.dataset_name == 'CheetahWorld-v2':
            s = traj['observations']
            a = traj['actions']
            r = traj['rewards']
            d = np.zeros(s.shape[0], dtype=np.int32)
            timesteps = np.arange(s.shape[0], dtype=np.int32)
            rtg = discount_cumsum(traj['rewards'], gamma=1.).reshape(-1, 1)
            task_idx = traj['idx']
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
            s = np.concatenate([np.zeros((self.max_len - tlen, self.state_dim), dtype=np.float32), s], axis=0) ## make s = (1, self.max_len, Ds)

            ## why use np.ones for action??
            a = np.concatenate([np.ones((self.max_len - tlen, self.act_dim), dtype=np.float32) * -10., a], axis=0) 


            r = np.concatenate([np.zeros((self.max_len - tlen, 1), dtype=np.float32), r], axis=0)

            if self.normalize:
                s = (s - self.state_mean) / self.state_std
                a = (a - self.action_mean) / self.action_std 
                r = (r - self.reward_mean) / self.reward_std

            d = np.concatenate([np.ones((self.max_len - tlen), dtype=np.int32) * 2, d], axis=0)
            rtg = np.concatenate([np.zeros((self.max_len - tlen, 1), dtype=np.float32), rtg], axis=0) / self.scale
            timesteps = np.concatenate([np.zeros((self.max_len - tlen), dtype=np.int32), timesteps], axis=0)
            mask = np.concatenate([np.zeros((self.max_len - tlen), dtype=np.float32), np.ones((tlen), dtype=np.float32)], axis=0)

        return [s, a, r, d, rtg, timesteps, mask], task_idx
