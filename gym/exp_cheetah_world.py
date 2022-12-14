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

from decision_transformer.dataset import eval_traj, get_trajectory_CheetahWorld, CustomDataset
from torch.utils.data import DataLoader

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
        ckpt_path = wandb.run.dir.split('/files')[0]
        print(f'\n{ckpt_path}\n')
    else:
        ckpt_path = None
    
    assert dataset_name == 'CheetahWorld-v2'
    trajectories = get_trajectory_CheetahWorld(env_name, env_level, variant['root'], dataset_name)
    
    max_ep_len = 200
    scale = 1.
    state_dim = 20
    act_dim = 6
    
    # save all path information into separate lists
    mode = variant.get('mode', 'normal')

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']

    device = variant.get('device', 'cuda')
    gpu_id = variant['gpu_id']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
 
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

    eval_fns = None

    assert model_type == 'de'

    mask_batch_fn = RandomPred(num_seqs=batch_size, seq_len=K, device=device)

    training_data = CustomDataset(dataset_name, env_name, env_level, 
        trajs=trajectories, max_len=K, eval_traj=eval_traj)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    trainer = MaskTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        scheduler=scheduler,
        eval_fns=eval_fns,
        ckpt_path=ckpt_path,
        mask_batch_fn=mask_batch_fn,
        train_dataloader=train_dataloader,
        device=device
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
    parser.add_argument('--log_to_wandb', '-w', type=boolean_argument, default=False)
    parser.add_argument('--root', type=str, default='./data', help='GPU id')

    args = parser.parse_args()

    experiment('gym', variant=vars(args))
