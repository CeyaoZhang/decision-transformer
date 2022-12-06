import numpy as np
import torch
import torch.nn as nn

from typing import Union, Dict

import transformers
from transformers import BertModel, BertForMaskedLM

class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])

# DecisionBERT model for zero_input mask
# where we use zero as the mask for input
# can be used for
# (1) element-level mask within state/action/reward
# (2) state/action/reward level mask
# (3) (s,a,r) transition level mask

class DecisionBERT(TrajectoryModel):
    """
    This model uses BERT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            input_type='cat',
            device='cuda',
            time_embed=False,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.input_type = input_type
        self.device=device
        self.time_embed = time_embed

        
        ## since we don't use input_ids but the input_embeds in the Bert, 
        # the word_embedding is useless and the vocab doesn't matter
        if self.input_type == 'seq':
            config = transformers.BertConfig(
                vocab_size=1,  
                hidden_size=hidden_size, 
                **kwargs
            )
        elif self.input_type == 'cat':
            config = transformers.BertConfig(
                vocab_size=1, 
                hidden_size=3*hidden_size, 
                **kwargs
            )

        ## self.bert = BertForMaskedLM(config) ## without use the pretrained model
        self.bert = BertModel(config, add_pooling_layer=False) # add_pooling_layer=False
        #print(f'total paras: {self.bert.num_parameters()}')
        #print('=' * 80)
        
        # self.embed_return = nn.Linear(1, hidden_size)
        self.embed_reward = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size) 
        self.embed_action = nn.Linear(self.act_dim, hidden_size)
        
        if self.input_type == 'seq':
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size) ## got it
            self.embed_ln = nn.LayerNorm(hidden_size)
            self.predict_state = nn.Linear(hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
            #self.predict_return = nn.Linear(hidden_size, 1)
            self.predict_reward = nn.Linear(hidden_size, 1)
            
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size).to(self.device))

        elif self.input_type == 'cat':
            self.embed_timestep = nn.Embedding(max_ep_len, 3*hidden_size) ## got it
            self.embed_ln = nn.LayerNorm(3*hidden_size)
            # self.predict_state = nn.Linear(3*hidden_size, self.state_dim)
            self.predict_state = nn.Linear(3*hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *([nn.Linear(3*hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
            #self.predict_return = nn.Linear(3*hidden_size, 1)
            self.predict_reward = nn.Linear(3*hidden_size, 1)
            
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 3*hidden_size).to(self.device))


    def forward(self, states, actions, rewards, 
                returns_to_go=None, timesteps=None, attention_mask=None, 
                return_outputs=False):
        '''
        states (B, L, Ds)
        attention_mask (B, L) if not None
        pooling: cls, mean, max
        '''

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for Bert: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.int32, device=self.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states) ## (B, L, Dh self.hidden_size)
        action_embeddings = self.embed_action(actions)
        # returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        
        ## this makes the sequence look like (s_1, a_1, r_1, s_2, a_2, r_2,...) (B, 3*seq, D)
        if self.input_type == 'seq':
            # time embeddings are treated similar to positional embeddings
            if self.time_embed:
                time_embeddings = self.embed_timestep(timesteps)
                state_embeddings = state_embeddings + time_embeddings
                action_embeddings = action_embeddings + time_embeddings
                # returns_embeddings = returns_embeddings + time_embeddings
                rewards_embeddings = rewards_embeddings + time_embeddings
        
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings, rewards_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size) 
            
            # append cls token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1) ## (Bs, 1, D)
            stacked_inputs = torch.cat((cls_tokens, stacked_inputs), dim=1) # bs * (1+3*seq) * dim
            ## after permute then reshape we can get (st, at, rt)
            stacked_inputs = self.embed_ln(stacked_inputs)

            ## here we think need to change
            # to make the attention mask fit the stacked inputs, have to stack it as well
            
            # append attention mask for cls_token
            ## (B, 3*L)
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3*seq_length) 
            ## (B, 1+3*L)
            stacked_attention_mask = torch.cat([torch.ones(batch_size, 1).to(self.device), stacked_attention_mask], dim=1)
            
        ## this makes the sequence look like (cat(s_1, a_1, r_1), cat(s_2, a_2, r_2), ...)
        elif self.input_type == 'cat':
            stacked_inputs = torch.concat(
                (state_embeddings, action_embeddings, rewards_embeddings), dim=2
            )
            if self.time_embed:
                time_embeddings = self.embed_timestep(timesteps)
                stacked_inputs = stacked_inputs + time_embeddings
            # append cls token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1) ## (B, 1, 3*D)
            stacked_inputs = torch.cat((cls_tokens, stacked_inputs), dim=1) # B * (1+seq) * (3*dim)
            stacked_inputs = self.embed_ln(stacked_inputs)
        
            # add mask for cls token
            ## (B, 1+L)
            stacked_attention_mask = torch.cat([torch.ones(batch_size, 1).to(self.device), attention_mask], dim=1)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.bert(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        # drop cls term
        outputs = transformer_outputs['last_hidden_state'][:, 1:] ## outputs [B,L,D]
        cls_output = transformer_outputs['last_hidden_state'][:,0] ## cls_output [B,D]
        # cls_output = transformer_outputs['pooler_output']

        if self.input_type == 'seq':
            ## outputs (B, 3*L, D)
            ## seq_outputs (B, 3, L, D)
            seq_outputs = outputs.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

            state_preds = self.predict_state(seq_outputs[:,0])     
            action_preds = self.predict_action(seq_outputs[:,1]) 
            reward_preds = self.predict_reward(seq_outputs[:,2])

        elif self.input_type == 'cat':
            ## outputs (B, L, 3*D) 
            ## direct pred s,a,r from the 3*D token
            state_preds = self.predict_state(outputs)     
            action_preds = self.predict_action(outputs) 
            reward_preds = self.predict_reward(outputs)

        if return_outputs:
            return (state_preds, action_preds, reward_preds), (outputs, cls_output)
        else:
            return state_preds, action_preds, reward_preds

    def get_traj_embedding(self, 
            outputs:torch.tensor, 
            cls_output:torch.tensor, 
            to_npy:bool=False)->Dict[str, Union[torch.tensor, np.array]]:

        # if pooling == 'cls':
        #     traj_embed = cls_output
        # elif pooling == 'mean':
        #     traj_embed = torch.mean(outputs, dim=1)
        # elif pooling == 'max':
        #     traj_embed = torch.max(outputs, dim=1)[0]
        traj_embed_cls = cls_output
        traj_embed_mean = torch.mean(outputs, dim=1)
        traj_embed_max = torch.max(outputs, dim=1)[0]
        traj_embed_mix = (traj_embed_cls+traj_embed_mean+traj_embed_max)/3

        if to_npy:
            return dict(
                    cls=traj_embed_cls.detach().cpu().numpy(), 
                    mean=traj_embed_mean.detach().cpu().numpy(), 
                    max=traj_embed_max.detach().cpu().numpy(),
                    mix=traj_embed_mix.detach().cpu().numpy()
                    )
        else:
            return dict(
                    cls=traj_embed_cls, 
                    mean=traj_embed_mean, 
                    max=traj_embed_max,
                    mix=traj_embed_mix
                    )

    

        


# where we use mask token as the mask for input
# can be used for
# (1) state/action/reward level mask
# (2) (s,a,r) transition level mask

class DecisionBERT_with_token_mask(TrajectoryModel):
    """
    This model uses BERT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            input_type='cat',
            random_mask=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.input_type = input_type

        # config = transformers.GPT2Config(
        #     vocab_size=1,  # doesn't matter -- we don't use the vocab
        #     n_embd=hidden_size,
        #     **kwargs
        # )
        if self.input_type == 'seq':
            config = transformers.BertConfig(
                vocab_size=1,  # doesn't matter -- we don't use the vocab
                hidden_size=hidden_size, 
                **kwargs
            )
        elif self.input_type == 'cat':
            config = transformers.BertConfig(
                vocab_size=1,  # doesn't matter -- we don't use the vocab
                hidden_size=3*hidden_size, 
                **kwargs
            )

        ## self.bert = BertForMaskedLM(config) ## without use the pretrained model
        self.bert = BertModel(config, add_pooling_layer=False)
        print(f'total paras: {self.bert.num_parameters()}')
        print('=' * 80)
        
        # self.embed_return = nn.Linear(1, hidden_size)
        self.embed_reward = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size) 
        self.embed_action = nn.Linear(self.act_dim, hidden_size)
        
        if self.input_type == 'seq':
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size) ## got it
            self.embed_ln = nn.LayerNorm(hidden_size)
            self.predict_state = nn.Linear(hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
            #self.predict_return = nn.Linear(hidden_size, 1)
            self.predict_reward = nn.Linear(hidden_size, 1)
            
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        elif self.input_type == 'cat':
            self.embed_timestep = nn.Embedding(max_ep_len, 3*hidden_size) ## got it
            self.embed_ln = nn.LayerNorm(3*hidden_size)
            self.predict_state = nn.Linear(3*hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *([nn.Linear(3*hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
            #self.predict_return = nn.Linear(hidden_size, 1)
            self.predict_reward = nn.Linear(3*hidden_size, 1)   
            
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 3*hidden_size))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 3*hidden_size))

    def random_token_mask(self, embedding):
        batch_size, seq_length, _ = embedding.shape
        # todo implement the function
        
        return masked_embedding
    
    def get_cls_output(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states) ## (batch_size, seq_length, self.hidden_size)
        action_embeddings = self.embed_action(actions)
        # returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)

        if self.input_type == 'seq':
            if self.time_embed:
                time_embeddings = self.embed_timestep(timesteps)
                state_embeddings = state_embeddings + time_embeddings
                action_embeddings = action_embeddings + time_embeddings
                # returns_embeddings = returns_embeddings + time_embeddings
                rewards_embeddings = rewards_embeddings + time_embeddings
        
            ## this makes the sequence look like (s_1, a_1, r_1, s_2, a_2, r_2,...)
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings, rewards_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size) 
            
            # append cls token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            stacked_inputs = torch.cat((cls_tokens, stacked_inputs), dim=1) # bs * (1+3*seq) * dim
            
            ## after permute then reshape we can get (st, at, rt)
            stacked_inputs = self.embed_ln(stacked_inputs)

            ## here we think need to change
            # to make the attention mask fit the stacked inputs, have to stack it as well
            # append attention mask for cls_token
            attention_mask = torch.cat([torch.ones(batch_size, 1).cuda(), attention_mask], dim=1)
            
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3*seq_length) 
            
            # we feed in the input embeddings (not word indices as in NLP) to the model
            transformer_outputs = self.bert(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
            )
            # drop the cls output
            cls_output = transformer_outputs['last_hidden_state'][:, 0]

        elif self.input_type == 'cat':
            ## this makes the sequence look like (cat(s_1, a_1, r_1), cat(s_2, a_2, r_2), ...)
            stacked_inputs = torch.concat(
                (state_embeddings, action_embeddings, rewards_embeddings), dim=2
            )
            if self.time_embed:
                time_embeddings = self.embed_timestep(timesteps)
                stacked_inputs = stacked_inputs + time_embeddings
                
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            stacked_inputs = torch.cat((cls_tokens, stacked_inputs), dim=1) # bs * (1+seq) * (3*dim)

            stacked_inputs = self.embed_ln(stacked_inputs)
            stacked_attention_mask = torch.cat([torch.ones(batch_size, 1).cuda(), attention_mask], dim=1)
        
            # we feed in the input embeddings (not word indices as in NLP) to the model
            transformer_outputs = self.bert(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
            )
            # drop the cls output
            cls_output = transformer_outputs['last_hidden_state'][:, 0]

        return cls_output
        
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, ):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states) ## (batch_size, seq_length, self.hidden_size)
        action_embeddings = self.embed_action(actions)
        # returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)

        if self.input_type == 'seq':
            if self.random_mask: # for training
                state_embeddings = self.random_token_mask(state_embeddings)
                action_embeddings = self.random_token_mask(action_embeddings)
                # returns_embeddings = returns_embeddings + time_embeddings
                rewards_embeddings = self.random_token_mask(rewards_embeddings)
            if self.time_embed:
                time_embeddings = self.embed_timestep(timesteps)
                state_embeddings = state_embeddings + time_embeddings
                action_embeddings = action_embeddings + time_embeddings
                # returns_embeddings = returns_embeddings + time_embeddings
                rewards_embeddings = rewards_embeddings + time_embeddings
        
            ## this makes the sequence look like (s_1, a_1, r_1, s_2, a_2, r_2,...)
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings, rewards_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size) 
            
            # append cls token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            stacked_inputs = torch.cat((cls_tokens, stacked_inputs), dim=1) # bs * (1+3*seq) * dim
            
            ## after permute then reshape we can get (st, at, rt)
            stacked_inputs = self.embed_ln(stacked_inputs)

            ## here we think need to change
            # to make the attention mask fit the stacked inputs, have to stack it as well
            # append attention mask for cls_token
            attention_mask = torch.cat([torch.ones(batch_size, 1).cuda(), attention_mask], dim=1)
            
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3*seq_length) 
            
            # we feed in the input embeddings (not word indices as in NLP) to the model
            transformer_outputs = self.bert(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
            )
            # drop the cls output
            x = transformer_outputs['last_hidden_state'][:, 1:]

            ## x (B, 3*L, D)
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

            state_preds = self.predict_state(x[:,0])     
            action_preds = self.predict_action(x[:,1]) 
            reward_preds = self.predict_reward(x[:,2])

        elif self.input_type == 'cat':
            ## this makes the sequence look like (cat(s_1, a_1, r_1), cat(s_2, a_2, r_2), ...)
            stacked_inputs = torch.concat(
                (state_embeddings, action_embeddings, rewards_embeddings), dim=2
            )
            if self.random_mask: # for training
                stacked_inputs = self.random_token_mask(stacked_inputs)
            if self.time_embed:
                time_embeddings = self.embed_timestep(timesteps)
                stacked_inputs = stacked_inputs + time_embeddings
                
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            stacked_inputs = torch.cat((cls_tokens, stacked_inputs), dim=1) # bs * (1+seq) * (3*dim)

            stacked_inputs = self.embed_ln(stacked_inputs)
            stacked_attention_mask = torch.cat([torch.ones(batch_size, 1).cuda(), attention_mask], dim=1)
        
            # we feed in the input embeddings (not word indices as in NLP) to the model
            transformer_outputs = self.bert(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
            )
            # drop the cls output
            x = transformer_outputs['last_hidden_state'][:, 1:]

            ## (B, L, 3*D) 
            #x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

            state_preds = self.predict_state(x)     
            action_preds = self.predict_action(x) 
            reward_preds = self.predict_reward(x)

        return state_preds, action_preds, reward_preds
    
import math
class PositionalEncoding(nn.Module):
    """Taken from the PyTorch transformers tutorial"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
