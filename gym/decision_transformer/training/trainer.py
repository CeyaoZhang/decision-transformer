import numpy as np
import torch

import time
import logging
logger = logging.getLogger(__name__)

import os 
import os.path as osp
PATH = os.getcwd()

class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, 
        scheduler=None, eval_fns=None, ckpt_path=None, loss_fn=None,
        ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.ckpt_path = PATH if ckpt_path is None else ckpt_path
        
        self.diagnostics = dict()
        self.start_time = time.time()
    
    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        model = self.model.module if hasattr(self.model, "module") else self.model
        # for param_key in model.state_dict():
        #     print(param_key, "\t", model.state_dict()[param_key].size())
        logger.info("saving %s", self.ckpt_path)
        torch.save(model.state_dict(), osp.join(self.ckpt_path, 'model.pth'))
    
    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = [] ## the loss in one iteration
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step() ## the loss in train one batch
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        
        ##
        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start
        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses) ## the mean in each iter
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
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
