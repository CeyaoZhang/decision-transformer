import gym
import numpy as np
import torch
import wandb

import argparse
import json
import random
import os
import sys
import datetime
import dateutil.tz

# from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.decision_bert import DecisionBERT ##
from decision_transformer.models.mlp_bc import MLPBCModel
# from decision_transformer.training.act_trainer import ActTrainer
# from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.training.trainer_distributed import Distributed_MaskTrainer ##
from decision_transformer.training.batches import RandomPred, BehaviorCloning, ForwardDynamics, BackwardsDynamics
# from decision_transformer.dataset import eval_traj, get_trajectory_CheetahWorld, CustomDataset
from decision_transformer.base_dataset import eval_traj, get_traj_from_dataset, CustomDataset

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from distutils.util import strtobool
def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10086" ## each exec should use different port
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

@torch.no_grad()
def eval_fn(
        eval_task_type:str,
        eval_dataloader,
        batch_size,
        K,
        device    
):
    def fn(model):
    
        if eval_task_type == 'Random':
            mask_batch_fn = RandomPred(num_seqs=batch_size, seq_len=K, device=device)
        elif eval_task_type == 'BC':
            mask_batch_fn = BehaviorCloning(num_seqs=batch_size, seq_len=K, device=device)
        elif eval_task_type == 'FD':
            mask_batch_fn = ForwardDynamics(num_seqs=batch_size, seq_len=K, device=device, rtg_masking_type='BC')
        elif eval_task_type == 'BD':
            mask_batch_fn = BackwardsDynamics(num_seqs=batch_size, seq_len=K, device=device, rtg_masking_type='BC')
        
        model.eval()
        eval_loss = 0
        for i, data in enumerate(eval_dataloader):

            features, task_idxs = data

            (states, actions, rewards, dones, \
                rtgs, timesteps, attention_masks) = features
            
            task_idxs = task_idxs.to(dtype=torch.int64, device=device)

            states = states.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.float32, device=device)
            rewards = rewards.to(dtype=torch.float32, device=device)
            dones = dones.to(dtype=torch.int32, device=device)
            rtgs = rtgs.to(dtype=torch.float32, device=device)
            timesteps = timesteps.to(dtype=torch.int32, device=device)
            attention_masks = attention_masks.to(dtype=torch.int32, device=device)

            input_masks = mask_batch_fn.input_masks
            state_inputs = states * input_masks["*"]["state"].unsqueeze(2) ## make input_masks from (B,L) to (B, L, 1) and will broadcast to states
            action_inputs = actions * input_masks["*"]["action"].unsqueeze(2)
            # reward_inputs = rtg[:,:-1] * input_masks["*"]["rtg"].unsqueeze(2)
            reward_inputs = rewards * input_masks["*"]["reward"].unsqueeze(2)

            assert state_inputs.shape[0] == rtgs.shape[0], 'please use the same length'
            (state_preds, action_preds, reward_preds), (outputs, cls_output) = model(
                state_inputs, action_inputs, reward_inputs, rtgs, timesteps, attention_masks,
                return_outputs=True
            )

            pred_masks = mask_batch_fn.get_prediction_masks()

            ## 
            state_preds_masks = pred_masks["*"]["state"].unsqueeze(2)
            state_preds = state_preds * state_preds_masks
            state_dim = state_preds.shape[2]
            ## concat the batch and length (B*L, D)
            state_preds = state_preds.reshape(-1, state_dim)[attention_masks.reshape(-1) > 0] 
            
            state_target = torch.clone(states * state_preds_masks)
            state_target = state_target.reshape(-1, state_dim)[attention_masks.reshape(-1) > 0]
            
            state_loss = torch.sum((state_preds - state_target)**2) / torch.sum(torch.abs(state_preds) > 0)

            ## 
            action_preds_masks = pred_masks["*"]["action"].unsqueeze(2)
            action_preds = action_preds * action_preds_masks
            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_masks.reshape(-1) > 0]
            
            action_target = torch.clone(actions * action_preds_masks)
            action_target = action_target.reshape(-1, act_dim)[attention_masks.reshape(-1) > 0]
            
            action_loss = torch.sum((action_preds - action_target)**2) / torch.sum(torch.abs(action_preds) > 0)

            reward_preds_masks = pred_masks["*"]["reward"].unsqueeze(2)
            reward_preds = reward_preds * reward_preds_masks
            reward_preds = reward_preds.reshape(-1, 1)[attention_masks.reshape(-1) > 0]
            
            reward_target = torch.clone(rewards * reward_preds_masks)
            reward_target = reward_target.reshape(-1, 1)[attention_masks.reshape(-1) > 0]
            
            reward_loss = torch.sum((reward_preds - reward_target)**2) / torch.sum(torch.abs(reward_preds) > 0)

            ##ss
            if eval_task_type == 'BC':
                print(f'step {i}: s {state_loss.detach().cpu().item()}, a {action_loss.detach().cpu().item()}, r {reward_loss.detach().cpu().item()}')
            
            masked_sar_loss = state_loss + action_loss + reward_loss
            eval_loss += masked_sar_loss.detach().cpu().item()
        
        return {
            f'{eval_task_type}': eval_loss
        }
    

    return fn


def experiment(
        rank,
        world_size,
        exp_prefix,
        variant,
):
    ddp_setup(rank, world_size)

    
    #device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)
    
    root = variant['root']
    dataset_name = variant['dataset']
    env_name, env_level = variant['env_name'], variant['env_level']
    model_type = variant['model_type']
    input_type = variant['input_type']
    train_task_type = variant['train_task_type']
    group_name = f'{exp_prefix}_{dataset_name}_{env_name}_{env_level}_{model_type}_{input_type}_{train_task_type}'
    if rank == 0:
        print(f'\ngroup_name: {group_name}\n')
    # exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S') # %y is 22 while %Y 2022
    exp_prefix = f'{group_name}_{timestamp}'
    
    max_ep_len = 200
    scale = 1.
    state_dim = 20
    act_dim = 6
    
    # save all path information into separate lists
    mode = variant.get('mode', 'normal')

    K = variant['K']
    batch_size = variant['batch_size']
    # num_eval_episodes = variant['num_eval_episodes']
    
    # dataset
    #training_data = CustomDataset(dataset_name, env_name, env_level, 
    #    trajs=trajectories, max_len=K, eval_traj=eval_traj)
    
    # get data
    # trajectories = get_trajectory_CheetahWorld(, variant['dataset'], variant['env_name'], variant['env_level'], variant['root'])
    trajectories, _ = get_traj_from_dataset(dataset_name, env_name, env_level, model_type, root)
    

    full_dataset = CustomDataset(dataset_name, env_name, env_level, trajs=trajectories, max_len=K, eval_traj=eval_traj, normalize=variant['normalize'])
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_dataloader = prepare_dataloader(train_dataset, batch_size)
    eval_dataloader = prepare_dataloader(eval_dataset, batch_size)

    if env_name=='all':
        num_task = 62
    elif env_name=='cheetah-vel':
        num_task = 60
    elif env_name=='cheetah-dir':
        num_task = 2
    
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
            device=rank,
            num_task=num_task,
        )
    else:
        raise NotImplementedError
    
    # optimizer and scheduler
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
    
    # get masked trainer
    mask_batch_fn = RandomPred(num_seqs=batch_size, seq_len=K, device=rank)
    # eval_tasks = ['Random', 'BC', 'FD', 'BD']
    eval_tasks = ['Random', 'BC']
    eval_fns = [eval_fn(eval_task, eval_dataloader, batch_size, K, rank) for eval_task in eval_tasks]


    trainer = Distributed_MaskTrainer(
        variant=variant,
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        gpu_id=rank,
        scheduler=scheduler,
        eval_fns=eval_fns,
        mask_batch_fn=mask_batch_fn,
    )


    if rank == 0:
        if log_to_wandb:
            wandb.init(
                name=exp_prefix,
                group=group_name,
                project='decision-bert', ## your project name
                config=variant,
                entity="porl" ## your wandb group name
            )
            # wandb.watch(model)  # wandb has some bug

            ## save the data info in order to provide normalize info to the downstream tasks
            save_path = os.path.join(wandb.run.dir, 'models')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            data_info_path = os.path.join(save_path, 'data_info.json')
            data_info = full_dataset.data_info
            data_info['task_embed_size'] = model.cls_token.shape[-1]
            with open(data_info_path, 'w') as f:
                json.dump(data_info, f, indent=4)

    trainer.train_iteration()
            
    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CheetahWorld-v2', 
                            choices=['D4RL', 'CheetahWorld-v2']) 
    parser.add_argument('--env_name', type=str, default='all') 
    parser.add_argument('--env_level', type=str, default='all')
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=200)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', '-bs', type=int, default=512)
    parser.add_argument('--model_type', type=str, default='de')  # dt for decision transformer, bc for behavior cloning, be for decision bert
    parser.add_argument('--embed_dim', type=int, default=160)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--activation_function', '-acf', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--save_epoch', type=int, default=20)
    parser.add_argument('--normalize', type=boolean_argument, default=True)
    parser.add_argument('--log_to_wandb', '-w', type=boolean_argument, default=False)
    parser.add_argument('--input_type', '-it', type=str, default='cat', choices=['seq', 'cat'], 
                            help='input tuples can be sequence type (s,a,r)+time  or concat type cat(s,a,r)') 
    
    parser.add_argument('--world_size', '-ws', type=int, default=8) # use how many gpus to distribute
    parser.add_argument('--root', type=str, default='/data/px/ceyaozhang/MyCodes/data', help='dataset path')
    
    args = parser.parse_args()
    
    world_size = args.world_size
    max_gpu_num = torch.cuda.device_count()
    print(f"max_gpu_num: {max_gpu_num}")
    assert world_size <= max_gpu_num, "The world size should not larger than the your device GPU number"
    
    import math
    args.learning_rate = args.learning_rate * math.ceil(np.sqrt(args.world_size))
    
    variant = vars(args)
    for (key, value) in variant.items():
        print(f"{key}: {value}")
    
    mp.spawn(experiment, args=(world_size, 'gym', variant), nprocs=world_size)
