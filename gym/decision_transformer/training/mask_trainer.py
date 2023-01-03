import numpy as np
import torch
from torch import linalg as LA
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss

from tqdm import tqdm
import wandb

import time

from decision_transformer.training.trainer import Trainer

class MaskTrainer(Trainer):

    def __init__(self, variant, model, optimizer, batch_size, 
        get_batch, scheduler, eval_fns, ckpt_path, 
        mask_batch_fn, train_dataloader, device
        ):

        super().__init__(model, optimizer, batch_size, 
            get_batch, scheduler, eval_fns, ckpt_path)
        # self.model = model
        # self.optimizer = optimizer
        # self.batch_size = batch_size
        # self.get_batch = get_batch
        self.scheduler = scheduler
        # self.eval_fns = [] if eval_fns is None else eval_fns
        self.variant = variant
        self.log_to_wandb = variant['log_to_wandb']
        self.b = variant['b'] ## the hyper balance two loss, default is 0.5
        self.pooling = variant['pooling']


        self.mask_batch_fn = mask_batch_fn
        self.train_dataloader = train_dataloader
        self.device = device

        self.size = len(train_dataloader.dataset)
        self.num_batch = len(train_dataloader)

        self.diagnostics = dict()
        self.start_time = time.time()
    
    def train_iteration(self):

        for epoch in tqdm(range(self.variant['epoch'])):
            
            train_losses = [] ## the loss in one iteration
            train_corrects = []
            epoch_logs = dict()

            train_start = time.time()
            self.model.train()
            for i, data in enumerate(self.train_dataloader):

                train_loss, correct  = self.train_step(data) ## the loss in train one batch
                train_losses.append(train_loss)
                train_corrects.append(correct)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            # get epoch logs
            epoch_time = time.time() - train_start
            epoch_logs['time/training'] = epoch_time

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
            acc = np.sum(train_corrects)/self.size 
            epoch_logs['training/train_acc'] = acc

            print(f'{epoch}: cost {epoch_time:.3f}s and acc {acc:.2f}')

            for k in self.diagnostics:
                epoch_logs[k] = self.diagnostics[k]


            if self.log_to_wandb:
                wandb.log(epoch_logs)

            # save the model params
            if (epoch+1) % self.variant['save_epoch'] == 0:
                self.save_checkpoint()
                print('=' * 80)
                print(f'save {epoch+1} model')

    def train_step(self, data):

        # states, actions, rewards, dones, \
        #     rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # (states, actions, rewards, dones, \
        #     rtgs, timesteps, attention_masks), _ = next(iter(self.train_dataloader))
        
        ## features []
        ## task_idxs torch.tensor [B]
        features, task_idxs = data

        (states, actions, rewards, dones, \
            rtgs, timesteps, attention_masks) = features

        task_idxs = task_idxs.to(dtype=torch.int64, device=self.device)

        states = states.to(dtype=torch.float32, device=self.device)
        actions= actions.to(dtype=torch.float32, device=self.device)
        rewards = rewards.to(dtype=torch.float32, device=self.device)
        dones = dones.to(dtype=torch.int32, device=self.device)
        rtgs = rtgs.to(dtype=torch.float32, device=self.device)
        timesteps = timesteps.to(dtype=torch.int32, device=self.device)
        attention_masks = attention_masks.to(dtype=torch.int32, device=self.device)
        
        ## mask the batch data
        ## both input_masks and pred_masks are (Batch, Length)
        ## pred_masks = 1 - input_masks
        ## those masked place (0 in input_masks) are where we need to pred (1 in pred_masks)
        # input_masks, pred_masks = self.mask_batch_fn.input_masks, self.mask_batch_fn.prediction_masks
        # input_masks, pred_masks = self.mask_batch_fn.get_input_masks(), self.mask_batch_fn.get_prediction_masks()
        input_masks = self.mask_batch_fn.input_masks

        state_inputs = states * input_masks["*"]["state"].unsqueeze(2) ## make input_masks from (B,L) to (B, L, 1) and will broadcast to states
        action_inputs = actions * input_masks["*"]["action"].unsqueeze(2)
        # reward_inputs = rtg[:,:-1] * input_masks["*"]["rtg"].unsqueeze(2)
        reward_inputs = rewards * input_masks["*"]["reward"].unsqueeze(2)

        ## use Bert to predict the mask token
        # state_preds, action_preds, reward_preds = self.model(
        #     state_inputs, action_inputs, reward_inputs, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        # )
        assert state_inputs.shape[0] == rtgs.shape[0], 'please use the same length'
        (state_preds, action_preds, reward_preds), (outputs, cls_output) = self.model(
            state_inputs, action_inputs, reward_inputs, rtgs, timesteps, attention_masks,
            return_outputs=True
        )
        # state_preds, action_preds, reward_preds = outputs['preds']
        
        ## task_embed [B, D]
        task_embed = self.model.get_traj_embedding(outputs, cls_output)[self.pooling]
        
        cls_task_loss, num_right_pred = self.cls_task_pred(task_embed, task_idxs)
        masked_sar_loss = self.masked_sar_pred(states, actions, rewards, attention_masks, state_preds, action_preds, reward_preds)
        # same_task_loss = self.same_task_pred(task_embed, task_idxs)
        # total_loss = (1.0-self.b) * masked_sar_loss + self.b * same_task_loss
        total_loss = (1.0-self.b) * masked_sar_loss + self.b * cls_task_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/masked_sar_error'] = masked_sar_loss.detach().cpu().item()
            # self.diagnostics['training/same_task_error'] = same_task_loss.detach().cpu().item()
            self.diagnostics['training/cls_task_error'] = cls_task_loss.detach().cpu().item()
            self.diagnostics['training/total_error'] = total_loss.detach().cpu().item()
        
        return total_loss.detach().cpu().item(), num_right_pred


    def masked_sar_pred(self, states, actions, rewards, attention_masks, state_preds, action_preds, reward_preds):
        pred_masks = self.mask_batch_fn.get_prediction_masks()

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

        # ##
        # rtg_preds = rtg_preds * pred_masks["*"]["rtg"].unsqueeze(2)
        # rtg_preds = rtg_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        # rtg_target = torch.clone(rtg_inputs)
        # rtg_target = rtg_target.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        # rtg_loss = torch.mean((rtg_preds - rtg_target)**2)

        ##
        reward_preds_masks = pred_masks["*"]["reward"].unsqueeze(2)
        reward_preds = reward_preds * reward_preds_masks
        reward_preds = reward_preds.reshape(-1, 1)[attention_masks.reshape(-1) > 0]
        
        reward_target = torch.clone(rewards * reward_preds_masks)
        reward_target = reward_target.reshape(-1, 1)[attention_masks.reshape(-1) > 0]
        
        reward_loss = torch.sum((reward_preds - reward_target)**2) / torch.sum(torch.abs(reward_preds) > 0)

        ##ss
        masked_sar_loss = state_loss + action_loss + reward_loss

        with torch.no_grad():
            self.diagnostics['training/state_error'] = state_loss.detach().cpu().item()
            self.diagnostics['training/action_error'] = action_loss.detach().cpu().item()
            # self.diagnostics['training/rtg_error'] = rtg_loss.detach().cpu().item()
            self.diagnostics['training/reward_error'] = reward_loss.detach().cpu().item()
        
        return masked_sar_loss

    
    def same_task_pred(self, task_embed, task_idxs)->torch.tensor:

        
        ## task_embed_norm [B, 1]
        task_embed_norm = LA.vector_norm(task_embed, dim=1, keepdim=True) 

        task_embed_pair_norm = task_embed_norm + task_embed_norm.T - 2*task_embed_norm.mm(task_embed_norm.T)
        task_id_pair = task_idxs.reshape(1, -1) == task_idxs.reshape(-1, 1)
        task_id_weight = task_id_pair*2 - 1.
        task_id_weight = task_id_weight.to(dtype=torch.float32, device=self.device)

        same_task_loss = task_embed_pair_norm.mul(task_id_weight).sum()

        return same_task_loss

    def cls_task_pred(self, task_embed:torch.tensor, task_idxs:torch.tensor)->torch.tensor:

        ## task_embed [B, D]
        ## task_score [B, #tasks]
        ## task_idxs [B]
        loss_fct = CrossEntropyLoss()
        task_score = self.model.cls(task_embed)
        cls_task_loss = loss_fct(task_score, task_idxs)
        correct  = (task_score.argmax(1) == task_idxs).type(torch.float).sum().item()

        return cls_task_loss, correct 

        



