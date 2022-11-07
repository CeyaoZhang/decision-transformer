import numpy as np
import torch
import torch.nn as nn

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
            timestep_encoding=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.timestep_encoding = timestep_encoding
        # config = transformers.GPT2Config(
        #     vocab_size=1,  # doesn't matter -- we don't use the vocab
        #     n_embd=hidden_size,
        #     **kwargs
        # )
        config = transformers.BertConfig(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            hidden_size=hidden_size, 
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        ## self.bert = BertForMaskedLM(config) ## without use the pretrained model
        self.bert = BertModel(config, add_pooling_layer=False)

        # Set up positional encoder
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=0.1) ## UniMASK mentioned use position encoder works better even for DT
        if self.timestep_encoding:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size) ## got it
        self.embed_return = nn.Linear(1, hidden_size)
        # self.embed_reward = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size) 
        self.embed_action = nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states) ## (batch_size, seq_length, self.hidden_size)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        # rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        if self.timestep_encoding:
        # time embeddings are treated similar to positional embeddings
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings
            returns_embeddings = returns_embeddings + time_embeddings
            # rewards_embeddings = rewards_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size) ## after permute then reshape we can get (rt, st, at)
        stacked_inputs = self.embed_ln(stacked_inputs)

        ## here we think need to change
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length) 

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.bert(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state'] ## why still last hidden state

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,0])  # predict return given return
        state_preds = self.predict_state(x[:,1])    # predict state given state 
        action_preds = self.predict_action(x[:,2])  # predict action given state

        return state_preds, action_preds, return_preds

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
