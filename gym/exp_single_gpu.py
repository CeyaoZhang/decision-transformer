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
from decision_transformer.training.mask_batches import RandomPred

from decision_transformer.base_dataset import CustomDataset, get_traj_from_dataset, eval_traj 
from torch.utils.data import DataLoader

from decision_transformer.evaluation.visualize_traj import VisualizeTraj

from distutils.util import strtobool
def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

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




def experiment(
        exp_prefix,
        variant,
):  
    print('=' * 50)
    for (key, value) in variant.items():
        print(f"{key}: {value}")
    

    dataset_name = variant['dataset']
    env_name, env_level = variant['env_name'], variant['env_level']
    model_type = variant['model_type']
    input_type = variant['input_type']
    group_name = f'{exp_prefix}_{dataset_name}_{env_name}_{env_level}_{model_type}_{input_type}'
    print(f'\ngroup_name: {group_name}\n')
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
    #     ckpt_path = wandb.run.dir.split('/files')[0]
    #     print(f'\n{ckpt_path}\n')
    
    # else:
    #     ckpt_path = None
    

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    # states, actions, rewards, returns, traj_lens, num_timesteps = eval_traj(env_name, env_level, trajectories, idx_name='all', mode=mode)

    # # used for input normalization
    # states = np.concatenate(states, axis=0) ## np.array (1M, ds)
    # state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    # actions = np.concatenate(actions, axis=0) ## np.array (1M, da)
    # action_mean, action_std = np.mean(actions, axis=0), np.std(actions, axis=0) + 1e-6
    # rewards = np.concatenate(rewards, axis=0) ## np.array (1M, 1)
    # reward_mean, reward_std = np.mean(rewards, axis=0), np.std(rewards, axis=0) + 1e-6


    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    ## this part works only pac_traj < 1.0, unless it's useless
    # only train on top pct_traj trajectories (for %BC experiment)
    # num_timesteps = max(int(pct_traj*num_timesteps), 1)
    # sorted_inds = np.argsort(returns)  # lowest to highest, give the order of each return
    # num_trajectories = 1
    # timesteps = traj_lens[sorted_inds[-1]]
    # ind = len(trajectories) - 2 ## 998
    # while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
    #     timesteps += traj_lens[sorted_inds[ind]]
    #     num_trajectories += 1
    #     ind -= 1
    # ## when terminal timesteps = 1M, num_trajectories=1000, ind = -1
    # sorted_inds = sorted_inds[-num_trajectories:]

    # # used to reweight sampling so we sample according to timesteps instead of trajectories
    # p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

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

    trajectories, (state_dim, act_dim, max_ep_len, env_targets) \
        = get_traj_from_dataset(dataset_name, env_name, env_level, model_type)
    
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

    train_type = variant['train_type']
    if train_type == 'pretrain':


        normalize=variant['normalize']
        training_data = CustomDataset(dataset_name, env_name, env_level, 
                trajs=trajectories, max_len=K, eval_traj=eval_traj, normalize=normalize)
                
        train_dataloader = DataLoader(training_data, batch_size=batch_size, 
                shuffle=True, num_workers=4, drop_last=True, pin_memory=True,) 

        if log_to_wandb:
            save_path = os.path.join(wandb.run.dir, 'models')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            data_info_path = os.path.join(save_path, f'data_info_{env_name}_{env_level}.json')
            data_info = training_data.data_info
            data_info['task_embed_size'] = model.cls_token.shape[-1]
            data_info['variant'] = variant
            with open(data_info_path, 'w') as f:
                json.dump(data_info, f, indent=4)  

        warmup_steps = variant['warmup_epochs']
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
                variant=variant,
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                eval_fns=eval_fns,
                ckpt_path=save_path,
                mask_batch_fn=mask_batch_fn,
                train_dataloader=train_dataloader,
                device=device
            )

        trainer.train_iteration()


    elif train_type == 'tSNE':

        # optionally load pre-trained weights
        if variant['path_to_weights'] != 'None':
            save_path = variant['path_to_weights']
            model_name = variant['model_name']
            model_path = osp.join(save_path, model_name)
            assert osp.exists(model_path), 'the model is not exists'
            model.load_state_dict(torch.load(model_path))
        else:
            save_path = './wandb/RandomBERT'
        
        training_data = CustomDataset(dataset_name, env_name, env_level, 
                trajs=trajectories, max_len=K, eval_traj=eval_traj)
        train_dataloader = DataLoader(
                training_data, 
                batch_size=400, 
                drop_last=True,
                shuffle=True, 
                num_workers=4
                )


        vis_traj = VisualizeTraj(train_dataloader, model, device, variant)
        # BERT_task_embedding = vistraj.get_task_embedding()
        tSNE_task_embedding = vis_traj.visualize(save_path)

        

            
    
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
    
    parser.add_argument('--train_type', type=str, default='pretrain', 
        choices=['pretrain', 'tSNE'], help='pretrain type: train a BERT model,\
                                        tSNE type: use a trained BERT model to visualize the task embedding')
    
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--model_type', type=str, default='dt',
                            choices=['dt', 'bc', 'de'], )  # dt for decision transformer, bc for behavior cloning, be for decision bert
    parser.add_argument('--input_type', '-it', type=str, default='cat', 
                            choices=['seq', 'cat'], 
                            help='input tuples can be sequence type (s,a,r)+time  or concat type cat(s,a,r)') 
    
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', '-acf', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    
    # parser.add_argument('--max_iters', type=int, default=10)
    # parser.add_argument('--num_steps_per_iter', '-ns', type=int, default=10000, help='how many batchs for training')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--K', type=int, default=200, help="max_traj_len")
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--num_eval_episodes', type=int, default=100)

    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--log_to_wandb', '-w', type=boolean_argument, default=False)

    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--normalize', type=boolean_argument, default=True)

    parser.add_argument('--path_to_weights', '-p2w', type=str, default=None, help='the path of pretrained model')
    parser.add_argument('--model_name', type=str, default='model.pth')
    # parser.add_argument('--pooling', type=str, default='cls', choices=['cls', 'mean', 'max'])

    args = parser.parse_args()

    experiment('gym', variant=vars(args))
