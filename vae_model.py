#%%
import torch
from torch import nn as nn
from torch.distributions import Normal, multivariate_normal
from torch.utils.data import Dataset, DataLoader
import numpy as np
import collections
from typing import Iterable
import scipy.sparse as sp_sparse
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from tqdm import trange
import sys
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scanpy as sc
import torch.nn.functional as F

#%%
device = torch.device("cpu")

#%%
class Generate_X:
    def __init__(self, p, give_mean = 'zero', give_sigma = 'one'):
        """
        give_sigma: whether the Sigma is diagonal matrix and same across cell types.
                    'one': The variances of genes are all ones.
                    'linspace': The variances of genes are different but even spaced numbers
                    'uniform': The variances of gene are randomly generated from [0,1]
        """
        self.p = p
        if give_mean is 'zero':
            self.mean = np.array([0]*self.p, dtype =float)
        elif give_mean is 'linspace':
            self.mean = np.linspace(0,2,num = self.p, dtype=float)
        elif give_mean is 'uniform':
            self.mean =  np.random.rand(self.p, dtype = float)
        else:
            print('Wrong argument!')
            pass
        if give_sigma is 'one':
            self.sigma = np.array([1]*self.p)
        elif give_sigma is 'linspace':
            self.sigma = np.linspace(0,2,num = self.p)
        elif give_mean is 'uniform':
            self.sigma =  np.random.rand(self.p)
        else:
            print('Wrong argument!')
            pass
        

    def sym_Sigma(self):
        p = self.p
        L = np.tril(np.random.random((p,p))*2,0)
        sparse = np.random.binomial(1,0.2,(p,p))
        diag_indices = np.diag_indices(p)
        sparse[diag_indices] = 1
        L = L * sparse
        S = np.matmul(L,L.transpose())
        return S

    def generate(self, n_ct, n_vec, same_mean = True, same_sigma = True):
        """
        n_ct: int, the number of cell types
        n_vec: the vector of cell size
        """
        assert n_ct == len(n_vec)
        ct_label = []
        rst = []
        covmat = []
        meanmat = []
        for ct in range(n_ct):  # ct: cell type
            if same_sigma:
                Sigma = np.diag(self.sigma)
            else:
                Sigma = self.sym_Sigma()
            Sigma = Sigma.astype(float)
            if same_mean:
                mean = self.mean
            else:
                mean = (np.random.rand(self.p)-0.5)*10
            MVN = np.random.multivariate_normal(mean, Sigma, n_vec[ct])
            #print(MVN)
            rst.append(MVN)
            ct_label.append(np.array([ct]*n_vec[ct]))
            covmat.append(Sigma)
            meanmat.append(mean)
        mat = np.array(rst)
        ct_label = np.array(ct_label)
        self.X = np.reshape(mat, (-1,self.p))
        self.ct_label = np.reshape(ct_label, (-1,))
        self.covmat = covmat
        self.meanmat = meanmat
        # print(self.X, self.ct_label)

#%%
def sim_sample_indices(sample_size):
    sample_indices = [[i]*sample_size[i] for i in range(len(sample_size))]
    return np.asarray(sample_indices).flatten()

#%%
class CytofDataset_Sim(Dataset):
    """ Create CyTOF dataset for Pytorch in Simulation
        The class contains __len__() and __getitem__() which could be used for dataloader.
    
    """

    def __init__(self,X, sample_indices, label_indices):
        self.dense = type(X) is np.ndarray
        self._X = np.ascontiguousarray(X, dtype=np.float32) if self.dense else X
        self.nb_genes = self.X.shape[1]
        self.samples, self.n_samples, self.sample_dict = arrange_categories(sample_indices)
        self.labels, self.n_labels, self.label_dict = arrange_categories(label_indices)

    @property
    def X(self):
        return self._X

    def __len__(self):
        return self._X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.samples[idx], self.labels[idx]
    
#    def collate_fn(self, batch):
#        indexes = np.array(batch)
#        X = self.X[indexes]
#        return self.collate_fn_end(X, indexes)
#    
#    def collate_fn_end(self, X, indexes):

#        if self.dense:
#            X = torch.from_numpy(X)
#        else:
#            X = torch.FloatTensor(X.toarray())
#        return X,  torch.LongTensor(self.samples[indexes]), \
#                   torch.LongTensor(self.labels[indexes])


    def get_sample(self, sample_sublist):
        """
        sample_sublist: list
        """
        loc = np.array([np.where(self.samples == self.sample_dict[sample]) for sample in sample_sublist]).flatten()
        X_sub = self._X[loc]
        samples_sub = self.samples[loc]
        labels_sub = self.labels[loc]
        return X_sub, samples_sub, labels_sub
    
    def get_celltype(self, celltype_sublist):
        """
        celltype_sublist: list
        """
        loc = np.array([np.where(self.labels == self.label_dict[batch]) for label in celltype_sublist]).flatten()
        X_sub = self._X[loc]
        samples_sub = self.samples[loc]
        labels_sub = self.labels[loc]
        return X_sub, samples_sub, labels_sub

#%%
def arrange_categories(original_categories, mapping_from=None, mapping_to=None):
    unique_categories = np.unique(original_categories)
    n_categories = len(unique_categories)
    batch_dict = dict()
    if mapping_to is None:
        mapping_to = range(n_categories)
    if mapping_from is None:
        mapping_from = unique_categories
    assert n_categories <= len(mapping_from)  # one cell_type can have no instance in dataset # Haven't debug it. Assume we have more than one cell type.
    assert len(mapping_to) == len(mapping_from)

    new_categories = np.copy(original_categories)
    for idx_from, idx_to in zip(mapping_from, mapping_to):
        # print(idx_from, idx_to)
        new_categories[original_categories == idx_from] = idx_to
        batch_dict[idx_from] = idx_to
    return new_categories.astype(int), n_categories, batch_dict

#%%
class Manual_Layers(nn.Module):
    def __init__(self, n_in=10, n_hidden=100, n_out=2):
        super(Manual_Layers, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(n_hidden, n_out*2),
        )
        
        
    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        
        return x


#%%
class Encoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_input: int, n_output: int,
                 n_layers: int = 1,
                 n_hidden: int = 128, dropout_rate: float = 0.1):
        super(Encoder, self).__init__()

        #self.encoder = FCLayers(n_in=n_input, n_out=n_hidden, n_layers=n_layers,
        #                        n_hidden=n_hidden, dropout_rate=dropout_rate)
        #self.mean_encoder = nn.Linear(n_hidden, n_output)
        #self.var_encoder = nn.Linear(n_hidden, n_output)
            
        self.encoder = Manual_Layers(n_in = n_input, n_hidden = n_hidden, n_out = n_output)
       

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()
    
    def encoder_function(self, x):
        h = self.encoder(x)
        q_m, q_logv = torch.chunk(h, 2, dim=1)
        return q_m, q_logv

    def forward(self, x: torch.Tensor):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q_m, q_logv = self.encoder_function(x)
        q_v = torch.exp(q_logv)
        
        #q = self.encoder(x)
        #q_m = self.mean_encoder(q)
        #q_v = torch.exp(self.var_encoder(q))  # (computational stability safeguard)torch.clamp(, -5, 5)
        latent = self.reparameterize(q_m, q_v)
        return q_m, q_v, latent

#%%
class DecoderVCSC(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space). It is also the number of genes
    :param n_cat_list: A list containing the number of categories
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    """

    def __init__(self, n_input: int, n_output: int,
                 n_layers: int = 1,
                 n_hidden: int = 128):
        super(DecoderVCSC, self).__init__()

        # shape: shape parameter in gamma distribution infered from estimated covariance matrix
        self.px_decoder =Manual_Layers(n_in = n_input, n_hidden = n_hidden, n_out = n_output)
        
        # mu
        self.px_mu_decoder =  nn.Linear(n_hidden, n_output)

#         # D
#         self.px_D_decoder = nn.Sequential(nn.Linear(n_hidden, n_output))

#         # L
#         self.px_L_decoder = nn.Linear(n_hidden, int(n_output*(n_output-1)/2))

#         # dropout
#         #self.px_dropout_decoder = nn.Linear(n_hidden, n_output)




    def forward(self, z: torch.Tensor):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the multiVariate normal distribution of expression

        :param z: tensor with shape ``(n_input,)``
        :param mu: empirical mean parameter in multivariate normal distribution with shape ``(#gene,)``
        :param D: diagonal vector of D in cholesky decomposition, where \Sigma = L^T * D * L, 
                    and \Sigma is cov matrix in multivariate normal distribution with shape ``(#gene,)``
        :param L: vector of unit lower triangular L matrix in cholesky decomposition with shape ``(#gene*(#gene-1)/2)``
        :return: parameters for the multivariate normal distribution of expression
        :rtype: `torch.Tensor`
        """

        

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z)
        px_mu = self.px_mu_decoder(px)
#         px_D = self.px_D_decoder(px)
#         px_D = torch.exp(px_D)
#         px_L = self.px_L_decoder(px)
        #print(px_L.size())
        #px_L = self.ReshapeL(px_L)
        #px_L = self.ReshapeL(px_L)
        #px_dropout = self.px_dropout_decoder(px_sigma)
        
        
#         return px_mu, px_D, px_L
        return px_mu

#%%
def log_gaussian_positive(x, px_mu):
    """
    This log likelihood is replaced by L2 square error
    """

    reconst_loss = F.binary_cross_entropy(px_mu, x, size_average=False)

    
    #res = part1 + part2
    return -reconst_loss


#%%
class VAE(nn.Module):
    r"""Variational auto-encoder model.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    """

    def __init__(self, n_input: int, n_labels: int = 18,
                 n_hidden: int = 128, n_latent: int = 10, n_layers: int = 1,
                 dropout_rate: float = 0.1, 
                 log_variational: bool = True):
        super(VAE, self).__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.n_labels = n_labels

        self.z_encoder = Encoder(n_input, n_latent, n_layers=n_layers, n_hidden=n_hidden,
                                 dropout_rate=dropout_rate)
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderVCSC(n_latent, n_input, n_layers=n_layers, n_hidden=n_hidden)

    def get_latents(self, x, give_mean):
        r""" returns the result of ``sample_from_posterior_z`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior_z(x, give_mean)]

    def sample_from_posterior_z(self, x, give_mean=False):
        r""" samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            pass
        else:
            x= exp(x)-1
        qz_m, qz_v, z = self.z_encoder(x)  
        if give_mean:
            z = qz_m
        return z

    def get_sample_mu(self, x):
        r"""Returns the tensor of predicted frequencies of expression

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :return: tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        """
        return self.inference(x)[0]

    # def get_sample_D(self, x):
    #     r"""Returns the tensor of means of the negative binomial distribution

    #     :param x: tensor of values with shape ``(batch_size, n_input)``
    #     :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
    #     """
    #     return self.inference(x)[1]

    # def get_sample_L(self, x):
    #     r"""Returns the tensor of means of the negative binomial distribution

    #     :param x: tensor of values with shape ``(batch_size, n_input)``
    #     :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
    #     """
    #     return self.inference(x)[2]

    def _reconstruction_loss(self, x, px_mu):
        # Reconstruction Loss
        #reconst_loss = -log_gaussian_positive(x, px_mu, px_D, px_L)
        #return reconst_loss
        
        part2 = log_gaussian_positive(x, px_mu)
        return -part2
    

    def inference(self, x):
        x_ = x
        if self.log_variational:
            pass
        else:
            x_ = torch.exp(x_) - 1

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_)

        # px_mu, px_D, px_L = self.decoder(z)
        px_mu = self.decoder(z)
        

        return px_mu, qz_m, qz_v, z
    
    # def reconst_x(self, px_mu, px_D):
    #     reconst_x = Normal(px_mu, 1/px_D).rsample()
        

    def forward(self, x):
        r""" Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :return: the reconstruction loss and the Kullback divergences
        """
        # Parameters for z latent distribution

        px_mu, qz_m, qz_v, z = self.inference(x)
        #reconst_loss = self._reconstruction_loss(x, px_mu, px_D, px_L)
        part2 = self._reconstruction_loss(x, px_mu)

        # KL Divergence between two multivariate normal distributions
        # mean = torch.zeros_like(qz_m)
        # scale = torch.ones_like(qz_v)

        # kl_divergence = torch.distributions.kl.kl_divergence(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=-1)
        kl_divergence = - 0.5 * torch.sum(1 + torch.log(qz_v) - qz_m**2 - qz_v)
        
        return part2, kl_divergence


        #return reconst_loss,  kl_divergence

#%%
class Trainer:
    """The abstract Trainer class for training a PyTorch model and monitoring its statistics. It should be
    inherited at least with a .loss() function to be optimized in the training loop.

    Args:
        :model: A model instance from class ``VAE``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
    """
    default_metrics_to_monitor = []
    

    def __init__(self, model, gene_dataset, benchmark=False,
                 verbose=False, kl=None, batch_size = 100):

        self.model = model
        self.gene_dataset = gene_dataset
        self.kl = kl
        self.verbose = verbose
        self.batch_size = batch_size

        self.benchmark = benchmark
        self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        self.training_time = 0
        
        self.history = []

        # self.gamma_index = gamma_index
        # self.cor_gt = cor_gt
        # self.cell_label = cell_label

    def data_loaders_loop(self):
        data_loader =  torch.utils.data.DataLoader(self.gene_dataset, batch_size = self.batch_size, shuffle =True)
        return data_loader

    def train(self, n_epochs=20, lr=1e-3, eps=0.01, params=None):
        begin = time.time()
        self.model.train()

        if params is None:
            params = filter(lambda p: p.requires_grad, self.model.parameters())

        # if hasattr(self, 'optimizer'):
        #     optimizer = self.optimizer
        # else:
        optimizer = self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps)  
        self.n_epochs = n_epochs

        with trange(n_epochs, desc="training", file=sys.stdout, disable=self.verbose) as pbar:
            # We have to use tqdm this way so it works in Jupyter notebook.
            # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)
                for tensors_list in self.data_loaders_loop():
                    X, sample_list, label_list = tensors_list
                    loss, part2 = self.loss(X)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    self.record_matrics(loss)
                #print("Epoch[{}/{}] Loss: {:.3f}".format(self.epoch, n_epochs, loss.data/self.batch_size))
                if self.epoch == 1:
                    self.init_z = self.get_latent(give_mean = True)
                    
        self.training_time += time.time() - begin
        print("\nTraining time:  %i s. / %i epochs" % (int(self.training_time), self.n_epochs))

    def loss(self, X):
        """
        ONLY for Debugging: The loss function = KL divergence + (x-p_mu)^2
        """
        #reconst_loss, kl_divergence = self.model(X)
        #reconst_loss, kl_divergence = self.model(X)
        part2, kl_divergence = self.model(X)
        #print(torch.mean(reconst_loss), torch.mean(self.kl_weight * kl_divergence))
        print('part2 = ', torch.mean(part2), 'KL = ', kl_divergence)
        #loss = torch.mean(reconst_loss + self.kl_weight * kl_divergence)
        #loss = torch.mean(reconst_loss)
        loss = torch.mean(part2 + self.kl_weight * kl_divergence)
        return loss, part2
    
    def record_matrics(self,loss):
        self.history += [loss]
        
    def plot_history(self):
        plt.plot(range(len(self.history)), self.history)


    def on_epoch_begin(self):
        self.kl_weight = self.kl if self.kl is not None else min(1, self.epoch / 400)  # self.n_epochs)
    
    @torch.no_grad()
    def get_latent(self, give_mean = True):
        x = self.gene_dataset.X
        x = torch.from_numpy(x).float()
        z = self.model.get_latents(x, give_mean)
        return z


#%%
# Generate Simulation dataset in 10 dimensional space and 2 cell types. There are 1500 cells in each cell type population. No multi-sample problem included.
sim1 = Generate_X(10)
sim1.generate(2,[1500,1500], same_mean = False, same_sigma = False)
sample_size = np.repeat(3000,1)
sample_indices = sim_sample_indices(sample_size)
#print(sample_indices)


#%%
# Generate CytofDataset_Sim class
simX_data = CytofDataset_Sim(sim1.X, sample_indices, sim1.ct_label)
simX, _, labels = simX_data.get_sample([0]) ## Used for simulation without multiple batches integration
gene_dataset = simX_data


#%%
# Parameters Setup
p = 10 # dimension of X
d = 2 # dimension of z
n_input = p
n_latent = d
n_layers = 3
n_hidden = 20 # dimension of hidden layer
dropout_rate = 0.1
n_epochs = 150
lr=1e-3
batch_size = 1500

#%%
np.savetxt('/Users/wenxuandeng/GoogleDrive/sucksalt/SC/vcsc/Version1/write/simulation', sim1.X, delimiter = ',')
adata = sc.read_csv('/Users/wenxuandeng/GoogleDrive/sucksalt/SC/vcsc/Version1/write/simulation')
adata.obs['cell type'] = sim1.ct_label
adata.obs['batch'] = sample_indices


#%%
sc.pp.neighbors(adata, n_neighbors = 15, metric = 'euclidean', random_state = None)
sc.tl.umap(adata, min_dist = 0.1)

#%%
sc.pl.umap(adata, color = 'cell type')

#%%
X_embedded = TSNE(n_components=2).fit_transform(gene_dataset.X)
#raw_plt = plt.scatter(X_embedded[:,0], X_embedded[:,1], c = gene_dataset.labels)
plt.scatter(X_embedded[:,0], X_embedded[:,1], c = gene_dataset.labels, s = [1.5]*len(X_embedded))
plt.savefig('/Users/wenxuandeng/GoogleDrive/sucksalt/SC/vcsc/Version1/figures/raw.80192019.3000.png')

#%%
vae = VAE(gene_dataset.nb_genes, gene_dataset.n_labels,
                 n_hidden = n_hidden, n_latent = n_latent, n_layers = n_layers)

#%%
# Initialization
mu_init = vae.get_sample_mu(torch.from_numpy(gene_dataset.X))


#%%
trainer = Trainer(vae, gene_dataset,  kl =  0.1, batch_size=batch_size)
trainer.train(n_epochs=n_epochs, lr=lr)

#%%
# Estimation after training
D_out = trainer.model.get_sample_D(torch.from_numpy(gene_dataset.X))
L_out = trainer.model.get_sample_L(torch.from_numpy(gene_dataset.X))
mu_out = trainer.model.get_sample_mu(torch.from_numpy(gene_dataset.X))

#%%
# Comparing true mean and estimated means
i= 0
plt.hist([mu_out[0:1500,i].detach().numpy(),mu_out[1500:3000,i].detach().numpy(), gene_dataset.X[0:1500,i], gene_dataset.X[1500:3000,i]], label = ["ct_1_out","ct_2_out", "ct_1", "ct_2"])


#%%
z = trainer.get_latent(give_mean =True)
z = z[0].numpy()
# Loss function curve with iterations
len(trainer.history)
trainer.plot_history()

#%%
plt.scatter(z[:,0], z[:,1], c = gene_dataset.labels, s = [.3]*len(z), marker='o')
plt.savefig('/Users/wenxuandeng/GoogleDrive/sucksalt/SC/vcsc/Version1/figures/vae.8192019.3000.png')

#%%
