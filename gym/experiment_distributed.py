import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import datetime
import dateutil.tz

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.decision_bert import DecisionBERT ##
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.training.trainer_distributed import Distributed_MaskTrainer ##
from decision_transformer.training.mask_batches import RandomPred

from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
import os

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8892"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def experiment(
        rank,
        world_size,
        exp_prefix,
        variant,
):
    ddp_setup(rank, world_size)
    
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    # exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S') # %y is 22 while %Y 2022
    exp_prefix = f'{group_name}-{timestamp}'
    
    # dataset todo
    dataset = get_dataset()
    
    # get data
    train_data = prepare_dataloader(dataset, variant['batch_size'])
    
    # get model
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
    
    # optimizer and scheduler
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
    
    # get masked trainer
    mask_batch_fn = RandomPred(num_seqs=variant['batch_size'], seq_len=K, device=device)
    trainer = Distributed_MaskTrainer(
        variant=variant,
        model=model,
        train_data=train_data
        optimizer=optimizer,
        gpu_id=rank,
        scheduler=scheduler,
        eval_fns=None,
        mask_batch_fn=mask_batch_fn,
    )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant,
            entity="porl" ## your wandb group name
        )
        # wandb.watch(model)  # wandb has some bug

    #for iter in range(variant['epoch']):
    trainer.train_iteration(variant)
            
    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper') # 'halfcheetah', 'hopper', 'walker2d'
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning, be for decision bert
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--input_type', '-it', type=str, default='cat', choices=['seq', 'cat'], 
                            help='input tuples can be sequence type (s,a,r)+time  or concat type cat(s,a,r)') 
    
    parser.add_argument('--world_size', type=int, default=2) # use how many gpus to distribute
    
    args = parser.parse_args()
    
    world_size = args.world_size
    mp.spawn(experiment, args=(world_size, 'gym', vars(args)), nprocs=world_size)
