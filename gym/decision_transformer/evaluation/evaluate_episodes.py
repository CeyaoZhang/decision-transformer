import numpy as np
import torch


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        ## why only add one zero token once a time not padding all at the beginning??
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(dtype=torch.float32, device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]

        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1) ## why
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1) ## why

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


from decision_transformer.training.batches import RandomPred, BehaviorCloning, ForwardDynamics, BackwardsDynamics

@torch.no_grad()
def eval_fn(
        eval_task_type:str,
        eval_dataloader,
        batch_size:int,
        K:int,
        device:str
        
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

            (states, actions, rewards, dones, rtgs, timesteps, attention_masks) = features
            
            task_idxs = task_idxs.to(dtype=torch.int64, device=device)

            states = states.to(dtype=torch.float32, device=device)
            actions= actions.to(dtype=torch.float32, device=device)
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
            
            # print(f'state: {state_preds.detach().cpu().tolist()}')
                
            ## 
            action_preds_masks = pred_masks["*"]["action"].unsqueeze(2)
            action_preds = action_preds * action_preds_masks
            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_masks.reshape(-1) > 0]
            
            action_target = torch.clone(actions * action_preds_masks)
            action_target = action_target.reshape(-1, act_dim)[attention_masks.reshape(-1) > 0]
            
            action_loss = torch.sum((action_preds - action_target)**2) / torch.sum(torch.abs(action_preds) > 0)
            
            # print(f'action: {action_preds} and {action_target}')
            
            ##
            reward_preds_masks = pred_masks["*"]["reward"].unsqueeze(2)
            reward_preds = reward_preds * reward_preds_masks
            reward_preds = reward_preds.reshape(-1, 1)[attention_masks.reshape(-1) > 0]
            
            reward_target = torch.clone(rewards * reward_preds_masks)
            reward_target = reward_target.reshape(-1, 1)[attention_masks.reshape(-1) > 0]
            
            reward_loss = torch.sum((reward_preds - reward_target)**2) / torch.sum(torch.abs(reward_preds) > 0)
            
            # print(f'reward: {reward_preds} and {reward_target}')

            ##ss
            # print(f'step {i}: s {state_loss.detach().cpu().item()}, a {action_loss.detach().cpu().item()}, r {reward_loss.detach().cpu().item()}')
            
            #     assert i != 0
            
            if eval_task_type == 'Random':
                masked_sar_loss = state_loss + action_loss + reward_loss
            elif eval_task_type == 'BC':
                masked_sar_loss = action_loss
            elif eval_task_type == 'FD' or eval_task_type == 'BD':
                masked_sar_loss = state_loss

            eval_loss += masked_sar_loss.detach().cpu().item()
        
        return {f'{eval_task_type}': eval_loss}
    
    return fn
        

