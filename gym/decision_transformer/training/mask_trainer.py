import numpy as np
import torch

import time

from decision_transformer.training.trainer import Trainer




class MaskTrainer(Trainer):

    def __init__(self, model, optimizer, 
        batch_size, get_batch, scheduler, eval_fns, mask_batch_fn
        ):

        super().__init__(model, optimizer, batch_size, get_batch, scheduler, eval_fns)
        # self.model = model
        # self.optimizer = optimizer
        # self.batch_size = batch_size
        # self.get_batch = get_batch
        # self.scheduler = scheduler
        # self.eval_fns = [] if eval_fns is None else eval_fns
        self.mask_batch_fn = mask_batch_fn

        self.diagnostics = dict()
        self.start_time = time.time()

    def train_step(self):

        states, actions, rewards, dones, \
            rtg, timesteps, attention_mask = self.get_batch(self.batch_size)

        ## mask the batch data
        ## both input_masks and pred_masks are (Batch, Length)
        input_masks, pred_masks = self.mask_batch_fn.input_masks, self.mask_batch_fn.prediction_masks
        # input_masks, pred_masks = self.mask_batch_fn.get_input_masks(), self.mask_batch_fn.get_prediction_masks()
        
        state_inputs = states * input_masks["*"]["state"].unsqueeze(2) ## make input_masks (B, L, 1) and will broadcast to states
        action_inputs = actions * input_masks["*"]["action"].unsqueeze(2)
        # reward_inputs = rtg[:,:-1] * input_masks["*"]["rtg"].unsqueeze(2)
        reward_inputs = rewards * input_masks["*"]["reward"].unsqueeze(2)

        ## use Bert to predict the mask token
        state_preds, action_preds, reward_preds = self.model(
            state_inputs, action_inputs, reward_inputs, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        ## 
        state_preds = state_preds * pred_masks["*"]["state"].unsqueeze(2)
        state_dim = state_preds.shape[2]
        state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0] ## concat the batch and length
        
        state_target = torch.clone(states * pred_masks["*"]["state"].unsqueeze(2))
        state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        
        state_loss = torch.mean((state_preds - state_target)**2)

        ## 
        action_preds = action_preds * pred_masks["*"]["action"].unsqueeze(2)
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        action_target = torch.clone(actions * pred_masks["*"]["action"].unsqueeze(2))
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        action_loss = torch.mean((action_preds - action_target)**2)

        # ##
        # rtg_preds = rtg_preds * pred_masks["*"]["rtg"].unsqueeze(2)
        # rtg_preds = rtg_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        # rtg_target = torch.clone(rtg_inputs)
        # rtg_target = rtg_target.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        # rtg_loss = torch.mean((rtg_preds - rtg_target)**2)

        ##
        reward_preds = reward_preds * pred_masks["*"]["rtg"].unsqueeze(2)
        reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        reward_target = torch.clone(rewards * pred_masks["*"]["reward"].unsqueeze(2))
        reward_target = reward_target.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        reward_loss = torch.mean((reward_preds - reward_preds)**2)

        ##
        # total_loss = state_loss + action_loss + rtg_loss
        total_loss = state_loss + action_loss + reward_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/state_error'] = state_loss.detach().cpu().item()
            self.diagnostics['training/action_error'] = action_loss.detach().cpu().item()
            # self.diagnostics['training/rtg_error'] = rtg_loss.detach().cpu().item()
            self.diagnostics['training/reward_error'] = reward_loss.detach().cpu().item()

        return total_loss.detach().cpu().item()


