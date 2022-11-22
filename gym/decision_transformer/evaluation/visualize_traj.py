import torch
import numpy as np
import pylab as plt

from tqdm import tqdm

from tsnecuda import TSNE as fTSNE
from sklearn.manifold import TSNE
import sys
sys.path.append('/data/px/ceyaozhang/OfficialCodes/FIt-SNE/')
from fast_tsne import fast_tsne


class VisualizeTraj():

    def __init__(self, dataloader, model, device) -> None:
        self.dataloader = dataloader
        self.device = device
        self.model = model.to(device)

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
        
        _, cls_output = self.model(states, actions, rewards,
             rtgs, timesteps, attention_masks, output_cls=True)

        return cls_output.detach().cpu().numpy(), task_idxs.detach().cpu().numpy()

    @torch.no_grad()
    def visualize(self):
        
        Xs, ys = [], []
        for _, data in enumerate(tqdm(self.dataloader)):
            X, y = self.get_task_embedding(data)
            Xs.append(X)
            ys.append(y)
        
        Xs = np.concatenate(Xs, axis=0)
        ys = np.concatenate(ys, axis=0)

        print(Xs.shape)
        # tsne = fTSNE(n_iter=5000, verbose=1, perplexity=10000, num_neighbors=500)
        # tsne = fTSNE(n_components=2, perplexity=15, learning_rate=10)
        tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
        
        # Zs = tsne.fit_transform(Xs)
        Zs = fast_tsne(Xs)

        print(Zs.shape)

        fig = plt.figure( figsize=(8,8) )
        ax = fig.add_subplot(1, 1, 1, title='TSNE' )

        # Create the scatter
        ax.scatter(x=Zs[:,0], y=Zs[:,1], c=ys, # label=y,
            cmap=plt.cm.get_cmap('Paired'), alpha=0.3, s=5.0)
        # ax.legend(loc='upper center', shadow=True)
        plt.savefig('temp.png')

        plt.show()

        return Zs

    