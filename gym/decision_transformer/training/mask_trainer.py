import numpy as np
import torch

import time


class MaskTrainer():

    def __init__(self, model, optimizer, 
        batch_size, get_batch, mask_batch_fn,
        scheduler=None, eval_fns=None
        ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.mask_batch_fn = mask_batch_fn

        self.diagnostics = dict()
        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs


    def train_step(self):

        states, actions, rewards, dones, \
            rtg, timesteps, attention_mask = self.get_batch(self.batch_size)

        ## mask the batch data
        ## both input_masks and pred_masks are (Batch, Length)
        input_masks, pred_masks = self.mask_batch_fn.input_masks, self.mask_batch_fn.prediction_masks
        
        states_inputs = states * input_masks["*"]["state"].unsqueeze(2) ## make input_masks (B, L, 1) and will broadcast to states
        actions_inputs = actions * input_masks["*"]["action"].unsqueeze(2)
        rtg_inputs = rtg[:,:-1] * input_masks["*"]["rtg"].unsqueeze(2)

        ## use Bert to predict the mask token
        state_preds, action_preds, rtg_preds = self.model(
            states_inputs, actions_inputs, rewards, rtg_inputs, timesteps, attention_mask=attention_mask,
        )

        ## 
        state_preds = state_preds * pred_masks["*"]["state"].unsqueeze(2)
        state_dim = state_preds.shape[2]
        state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        
        state_target = torch.clone(states_inputs)
        state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        
        state_loss = torch.mean((state_preds - state_target)**2)

        ## 
        action_preds = action_preds * pred_masks["*"]["action"].unsqueeze(2)
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        action_target = torch.clone(actions_inputs)
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        action_loss = torch.mean((action_preds - action_target)**2)

        ##
        rtg_preds = rtg_preds * pred_masks["*"]["rtg"].unsqueeze(2)
        rtg_preds = rtg_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        rtg_target = torch.clone(rtg_inputs)
        rtg_target = rtg_target.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        rtg_loss = torch.mean((rtg_preds - rtg_target)**2)

        ##
        total_loss = state_loss + action_loss + rtg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/state_error'] = state_loss.detach().cpu().item()
            self.diagnostics['training/action_error'] = action_loss.detach().cpu().item()
            self.diagnostics['training/rtg_error'] = rtg_loss.detach().cpu().item()

        return total_loss.detach().cpu().item()


