# import gym
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
# import re


from typing import List, Dict

# from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.decision_bert import DecisionBERT ##
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.training.mask_trainer import MaskTrainer ##
from decision_transformer.training.batches import RandomPred, BehaviorCloning, ForwardDynamics, BackwardsDynamics

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

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    ## if you add new keys in the json file, please add the same keys in the default.py
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

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
            mask_batch_fn = ForwardDynamics(num_seqs=batch_size, seq_len=K, device=device)
        elif eval_task_type == 'BD':
            mask_batch_fn = BackwardsDynamics(num_seqs=batch_size, seq_len=K, device=device)
        
        model.eval()
        eval_loss = 0
        for i, data in enumerate(eval_dataloader):

            features, task_idxs = data

            (states, actions, rewards, dones, \
                rtgs, timesteps, attention_masks) = features
            
            task_idxs = task_idxs.to(dtype=torch.int64, device=device)

            states = states.to(dtype=torch.float32, device=device)
            actions= actions.to(dtype=torch.float32, device=device)
            rewards = rewards.to(dtype=torch.float32, devie=device)
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
            masked_sar_loss = state_loss + action_loss + reward_loss

            eval_loss += masked_sar_loss.detach().cpu().item()
        
        return {
            f'{eval_task_type}': eval_loss
        }
    

    return fn


def experiment(
        exp_prefix,
        variant,
):  

    print('=' * 50)
    for (key, value) in variant.items():
        print(f"{key}: {value}")
    
    root = variant['root']
    dataset_name = variant['dataset']
    env_name, env_level = variant['env_name'], variant['env_level']
    model_type = variant['model_type']
    input_type = variant['input_type']
    train_task_type = variant['train_task_type']
    group_name = f'{exp_prefix}_{dataset_name}_{env_name}_{env_level}_{model_type}_{input_type}_{train_task_type}'
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


    device = variant.get('device', 'cuda')
    gpu_id = variant['gpu_id']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    trajectories, (state_dim, act_dim, max_ep_len, env_targets) \
        = get_traj_from_dataset(dataset_name, env_name, env_level, model_type, root)
    
    if env_name=='all':
        num_task = 62
    elif env_name=='cheetah-vel':
        num_task = 60
    elif env_name=='cheetah-dir':
        num_task = 2
    
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
            device=device,
            num_task=num_task,
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    train_type = variant['train_type']
    if train_type == 'pretrain':


        normalize=variant['normalize']
        full_dataset = CustomDataset(dataset_name, env_name, env_level, 
                trajs=trajectories, max_len=K, eval_traj=eval_traj, normalize=normalize)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
                
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                shuffle=True, num_workers=4, drop_last=True, pin_memory=True,) 
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, 
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
        else:
            save_path = None

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
            # eval_fns = [eval_episodes(tar) for tar in env_targets]
            pass
        elif dataset_name == 'CheetahWorld-v2':
            eval_fns = None
            get_batch = None

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
            if train_task_type == 'Random':
                mask_batch_fn = RandomPred(num_seqs=batch_size, seq_len=K, device=device)
            elif train_task_type == 'BC':
                mask_batch_fn = BehaviorCloning(num_seqs=batch_size, seq_len=K, device=device)
            
            eval_tasks = ['Random', 'BC', 'FD', 'BD']
            eval_fns = [eval_fn(eval_task, eval_dataloader, batch_size,K, device) for eval_task in eval_tasks]

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
                device=device,
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
    
    parser.add_argument('--train_type', type=str, default='pretrain', 
        choices=['pretrain', 'tSNE'], help='pretrain type: train a BERT model,\
                                        tSNE type: use a trained BERT model to visualize the task embedding')
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--model_type', type=str, default='dt',
                            choices=['dt', 'bc', 'de'], )  # dt for decision transformer, bc for behavior cloning, be for decision bert
    parser.add_argument('--input_type', '-it', type=str, default='cat', 
                            choices=['seq', 'cat'], 
                            help='input tuples can be sequence type (s,a,r)+time  or concat type cat(s,a,r)') 
    parser.add_argument('--train_task_type', type=str, default='Random',
                            choices=['Random', 'BC'], )  


    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', '-acf', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--b', type=float, default=0.5, help='a hyperparameter balance two losses')

    
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
    parser.add_argument('--pooling', type=str, default='cls', choices=['cls', 'mean', 'max', 'mix'])

    parser.add_argument('--root', type=str, default='/data/px/ceyaozhang/MyCodes/data', help='dataset path')

    args = parser.parse_args()

    experiment('gym', variant=vars(args))
