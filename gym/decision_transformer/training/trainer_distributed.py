import numpy as np
import torch

import time
from torch.nn.parallel import DistributedDataParallel as DDP
import os

class Distributed_MaskTrainer:

    def __init__(self, variant, model, train_data, optimizer, gpu_id,
        scheduler=None, eval_fns=None, loss_fn=None,
        ):
        self.variant = varaint
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        
        self.optimizer = optimizer
        self.train_data = train_data

        self.loss_fn = loss_fn
        
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        
        self.diagnostics = dict()
        self.start_time = time.time()
        
        self.model = DDP(model, device_ids=[gpu_id])
        
        log_to_wandb = variant.get('log_to_wandb', False)
        self.log_to_wandb = log_to_wandb
        
    def train_iteration(self):

        logs = dict()
        epoch_logs = dict()
        
        train_step = 0 
        self.model.train()

        for epoch in range(self.variant['epoch']):
            self.train_data.sampler.set_epoch(epoch)
            train_losses = [] ## the loss in one iteration
            train_start = time.time()
            for i, data in enumerate(train_loader):
                self.train_step(data) ## the loss in one step
                train_losses.append(logs['training/total_error'])
                
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                for k in self.diagnostics:
                    logs[k] = self.diagnostics[k]
                    
                if self.log_to_wandb:
                    wandb.log(outputs, step=train_step)
                    
                train_step += 1
        
            # get epoch logs
            epoch_logs['time/training'] = time.time() - train_start

            eval_start = time.time()

            # evaluate models
            self.model.eval()
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model)
                for k, v in outputs.items():
                    epoch_logs[f'evaluation/{k}'] = v

            epoch_logs['time/total'] = time.time() - self.start_time
            epoch_logs['time/evaluation'] = time.time() - eval_start
            epoch_logs['training/train_loss_mean'] = np.mean(train_losses) ## the mean in each iter
            epoch_logs['training/train_loss_std'] = np.std(train_losses)
            
            if self.log_to_wandb:
                wandb.log(epoch_logs, step=epoch)
            
            print('=' * 80)
            print(f'Iteration {epoch}')
            for k, v in epoch_logs.items():
                print(f'{k}: {v}')

    def train_step(self, data):
        
        states, actions, rewards, dones, \
            rtg, timesteps, attention_mask = data
        
        input_masks, pred_masks = self.mask_batch_fn.get_input_masks(), self.mask_batch_fn.get_prediction_masks()
        
        state_inputs = states * input_masks["*"]["state"].unsqueeze(2) ## make input_masks from (B,L) to (B, L, 1) and will broadcast to states
        action_inputs = actions * input_masks["*"]["action"].unsqueeze(2)
        reward_inputs = rewards * input_masks["*"]["reward"].unsqueeze(2)

        state_preds, action_preds, reward_preds = self.model(
            state_inputs, action_inputs, reward_inputs, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        state_preds_masks = pred_masks["*"]["state"].unsqueeze(2)
        state_preds = state_preds * state_preds_masks
        state_dim = state_preds.shape[2]
        ## concat the batch and length (B*L, D)
        state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0] 
        
        state_target = torch.clone(states * state_preds_masks)
        state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        
        state_loss = torch.sum((state_preds - state_target)**2) / torch.sum(state_preds_masks)

        action_preds_masks = pred_masks["*"]["action"].unsqueeze(2)
        action_preds = action_preds * action_preds_masks
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        action_target = torch.clone(actions * action_preds_masks)
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        action_loss = torch.sum((action_preds - action_target)**2) / torch.sum(action_preds_masks)

        reward_preds_masks = pred_masks["*"]["reward"].unsqueeze(2)
        reward_preds = reward_preds * reward_preds_masks
        reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        reward_target = torch.clone(rewards * reward_preds_masks)
        reward_target = reward_target.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        reward_loss = torch.sum((reward_preds - reward_target)**2) / torch.sum(reward_preds_masks)
        
        total_loss = state_loss + action_loss + reward_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            self.diagnostics['training/state_error'] = state_loss.detach().cpu().item()
            self.diagnostics['training/action_error'] = action_loss.detach().cpu().item()
            self.diagnostics['training/reward_error'] = reward_loss.detach().cpu().item()
            self.diagnostics['training/total_error'] = total_loss.detach().cpu().item()
