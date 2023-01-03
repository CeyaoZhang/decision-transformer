import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import json
import random
import sys
import datetime
import dateutil.tz
import os.path as osp
import os 
import re


from typing import List, Dict

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.decision_bert import DecisionBERT ##
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.training.mask_trainer import MaskTrainer ##
from decision_transformer.training.batches import RandomPred

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def get_mean_std(x:List[np.array]) -> np.array:
    x = np.concatenate(x, axis=0) ## np.array (1M, Ds)
    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0) + 1e-6
    return x_mean, x_std

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



def experiment(
        exp_prefix,
        variant,
):  
    for (key, value) in variant.items():
        print(f"{key}: {value}")
    
    dataset_name = variant['dataset']
    env_name, env_level = variant['env_name'], variant['env_level']
    model_type = variant['model_type']
    input_type = variant['input_type']
    group_name = f'{exp_prefix}_{dataset_name}_{env_name}_{env_level}_{model_type}_{input_type}'
    # exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S') # %y is 22 while %Y 2022
    exp_prefix = f'{group_name}_{timestamp}'

    log_to_wandb = variant.get('log_to_wandb', False)
    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-bert',
            config=variant,
            entity="porl" ## your wandb group name
        )
        # wandb.watch(model)  # wandb has some bug
        ckpt_path = wandb.run.dir.split('/files')[0]
        print(f'\n{ckpt_path}\n')
    else:
        ckpt_path = None
    
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
    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, actions, rewards, returns, traj_lens, num_timesteps = eval_traj(env_name, env_level, trajectories, idx_name='all', mode=mode)

    # used for input normalization
    states = np.concatenate(states, axis=0) ## np.array (1M, ds)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    actions = np.concatenate(actions, axis=0) ## np.array (1M, da)
    action_mean, action_std = np.mean(actions, axis=0), np.std(actions, axis=0) + 1e-6
    rewards = np.concatenate(rewards, axis=0) ## np.array (1M, 1)
    reward_mean, reward_std = np.mean(rewards, axis=0), np.std(rewards, axis=0) + 1e-6


    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    ## this part works only pac_traj < 1.0, unless it's useless
    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest, give the order of each return
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2 ## 998
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    ## when terminal timesteps = 1M, num_trajectories=1000, ind = -1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    device = variant.get('device', 'cuda')
    gpu_id = variant['gpu_id']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            ## all the data from here !! but why we need sorted_inds??
            traj = trajectories[int(sorted_inds[batch_inds[i]])] 

            if dataset_name == 'D4RL':
                si = random.randint(0, traj['rewards'].shape[0] - 1)
                # get sequences from dataset
                ## from (max_len, d) to (1, max_len, d) for s, a, r
                ## from (max_len, ) to (1, max_len) for terminals, timesteps
                s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim)) 
                a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
                r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
                if 'terminals' in traj:
                    d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
                elif 'dones' in traj:
                    d.append(traj['dones'][si:si + max_len].reshape(1, -1))
                ## why not se si+max_len, because si + max_len may > len(traj)
                timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1)) 
                timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
                ## len(rtg[0]) = len(r[0]) + 1, but I don't know the usage
                rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
                if rtg[-1].shape[1] <= s[-1].shape[1]:
                    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            elif dataset_name == 'CheetahWorld-v2':
                s.append(traj['observations'].reshape(1, -1, state_dim)) 
                a.append(traj['actions'].reshape(1, -1, act_dim))
                r.append(traj['rewards'].reshape(1, -1, 1))
                d.append(np.zeros(s[-1].shape[1]).reshape(1, -1)) 
                timesteps.append(np.arange(s[-1].shape[1]).reshape(1, -1)) 
                rtg.append(discount_cumsum(traj['rewards'], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
                if rtg[-1].shape[1] <= s[-1].shape[1]:
                    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding, state + reward normalization
            ## at first glance, this is strange for padding from the left,
            ## but then I realized, that it doesn't matter from the left or the right,
            ## the key point is to be consistence with mask
            tlen = s[-1].shape[1] ## the len of each traj
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1) ## make s[-1] = (1, max_len, Ds)
            s[-1] = (s[-1] - state_mean) / state_std
            ## why use np.ones for action??
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1) 
            a[-1] = (a[-1] - action_mean) / action_std 
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            r[-1] = (r[-1] - reward_mean) / reward_std 
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1)) 

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.int, device=device) ## default version is torch.long
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.int, device=device) ## default version is torch.long
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.float32, device=device) ## default version without torch.float32

        ## s,a,r,rtg are (B, L, D), d and mask are (B,L)
        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    elif model_type == 'bc':
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    
    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    elif model_type == 'de':
        model = DecisionBERT(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            num_hidden_layers=variant['n_layer'],
            num_attention_heads=variant['n_head'],
            intermediate_size=4*variant['embed_dim'],
            hidden_act=variant['activation_function'],
            max_position_embeddings=1024,
            hidden_dropout_prob=variant['dropout'],
            attention_probs_dropout_prob=variant['dropout'],
            input_type=variant['input_type'],
            device=device
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if dataset_name == 'D4RL':
        eval_fns = [eval_episodes(tar) for tar in env_targets]
    elif dataset_name == 'CheetahWorld-v2':
        eval_fns = None

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=eval_fns,
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=eval_fns,
        )
    elif model_type == 'de':
        mask_batch_fn = RandomPred(num_seqs=batch_size, seq_len=K, device=device)
        trainer = MaskTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            eval_fns=eval_fns,
            ckpt_path=ckpt_path,
            mask_batch_fn=mask_batch_fn,
        )

    

    for iter in range(variant['max_iters']):
        logs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(logs)
        
        ## save the model params
        if (iter+1) % 10 == 0:
            trainer.save_checkpoint()
            print(f'save model')

    ## save the Bert model for 
    # model = 
    # PATH = './model.pth'
    # model.load_state_dict(torch.load(PATH))
            
    
    print("\n---------------------Finish!!!----------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='D4RL', 
                            choices=['D4RL', 'CheetahWorld-v2']) 
    parser.add_argument('--env_name', type=str) 
    parser.add_argument('--env_level', type=str)
    
    # subparsers = parser.add_subparsers()
    
    # parser_d4rl = subparsers.add_parser('D4RL', help='D4RL dataset')
    # parser_d4rl.add_argument('--env_name', type=str, default='hopper', choices=['halfcheetah', 'hopper', 'walker2d']) # 
    # parser_d4rl.add_argument('--env_level', type=str, default='medium', choices=['medium', 'medium-replay', 'medium-expert', 'expert'])  
    
    # parser_cheetah = subparsers.add_parser('CheetahWorld-v2', help='CheetahWorld-v2 dataset')
    # parser_cheetah.add_argument('--env_name', type=str, default='cheetah-dir') 
    # parser_cheetah.add_argument('--env_level', type=str, default='normal')
    
    
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning, be for decision bert
    parser.add_argument('--input_type', '-it', type=str, default='cat', 
                            choices=['seq', 'cat'], 
                            help='input tuples can be sequence type (s,a,r)+time  or concat type cat(s,a,r)') 
    
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000, help='how many batchs for training')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--num_eval_episodes', type=int, default=100)

    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)

    args = parser.parse_args()

    experiment('gym', variant=vars(args))
