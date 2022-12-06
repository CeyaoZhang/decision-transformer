import torch
import numpy as np
import pylab as plt

import os
from time import time

from tqdm import tqdm

from tsnecuda import TSNE as fTSNE
from sklearn.manifold import TSNE
import sys
sys.path.append('/data/px/ceyaozhang/OfficialCodes/FIt-SNE/')
from fast_tsne import fast_tsne


class VisualizeTraj():

    def __init__(self, dataloader, model, device, variant:str):
        self.dataloader = dataloader
        self.device = device
        self.model = model.to(device)
        self.variant = variant

    def get_task_embedding(self, data):
        features, task_idxs = data

        (states, actions, rewards, dones, \
            rtgs, timesteps, attention_masks) = features
        states = states.to(dtype=torch.float32, device=self.device)
        actions= actions.to(dtype=torch.float32, device=self.device)
        rewards = rewards.to(dtype=torch.float32, device=self.device)
        dones = dones.to(dtype=torch.int32, device=self.device)
        rtgs = rtgs.to(dtype=torch.float32, device=self.device)
        timesteps = timesteps.to(dtype=torch.int32, device=self.device)
        attention_masks = attention_masks.to(dtype=torch.float32, device=self.device)
        
        _, (outputs, cls_output) = self.model(states, actions, rewards,
             rtgs, timesteps, attention_masks, return_outputs=True)
        traj_embed_dict = self.model.get_traj_embedding(outputs, cls_output, to_npy=True)

        return traj_embed_dict, task_idxs.detach().cpu().numpy()

    @torch.no_grad()
    def visualize(self, save_path):
        
        # if save_path == None:
        #     save_path = self.variant['path_to_weights']
        # assert save_path != None, "a save path must be given!"

        tsne_path = os.path.join(save_path, 'tsne')
        if not os.path.exists(tsne_path):
            os.makedirs(tsne_path)
        dataset_name = self.variant['dataset']
        env_name, env_level = self.variant['env_name'], self.variant['env_level']
        


        Xs_cls, Xs_mean, Xs_max, Xs_mix, ys = [], [], [], [], []
        ## Due to device resource limitations, use batch to load all data 
        for _, data in enumerate(tqdm(self.dataloader)):
            feat, y = self.get_task_embedding(data)
            Xs_cls.append(feat['cls'])
            Xs_mean.append(feat['mean'])
            Xs_max.append(feat['max'])
            Xs_mix.append(feat['mix'])
            ys.append(y)
        Xs_cls = np.concatenate(Xs_cls, axis=0)
        Xs_mean = np.concatenate(Xs_mean, axis=0)
        Xs_max = np.concatenate(Xs_max, axis=0)
        Xs_mix = np.concatenate(Xs_mix, axis=0)
        ys = np.concatenate(ys, axis=0)

        print(Xs_cls.shape, ys.shape)
        idx = np.arange(ys.shape[0])
        np.random.shuffle(idx)
        
        
        poolings = ['cls', 'mean', 'max', 'mix']
        inputs = [Xs_cls, Xs_mean, Xs_max, Xs_mix]
        outputs = []
        for (pooling, input) in zip(poolings, inputs):
            pic_name = f'{env_name}_{env_level}_{pooling}'
            pic_path = os.path.join(tsne_path, pic_name)
            
            # tsne = fTSNE(n_iter=5000, verbose=1, perplexity=10000, num_neighbors=500)
            tsne = fTSNE(n_components=2, perplexity=15, learning_rate=10)
            # tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
            
            start_time = time()
            Z = tsne.fit_transform(input[idx[:50000]])
            outputs.append(Z)
            # Zs = fast_tsne(Xs)
            end_time = time()
            print(f'tsne result {Z.shape} and cost {end_time-start_time:.2}s')

            fig = plt.figure( figsize=(8,8) )
            ax = fig.add_subplot(1, 1, 1, title=pic_name)

            # Create the scatter
            ax.scatter(x=Z[:,0], y=Z[:,1], s=2.0, c=ys[idx[:50000]], alpha=0.5,# label=y,
                cmap=plt.cm.get_cmap('Paired'))
            # ax.legend(loc='upper center', shadow=True)
            
            plt.savefig(pic_path+f'_{int(end_time)}.png')
            plt.show()

        print(f'\n{env_name} | { env_level} Done!')

        return outputs

    