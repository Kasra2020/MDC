# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:58:16 2021

@author: Kasra Rafiezadeh Shahi
"""


import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import _supervised
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing
import torch
import torch.nn as nn



# =============================================================================
# Clustering Accuracy (CA)
# =============================================================================
def clustering_accuracy(labels_true, labels_pred):
    labels_true, labels_pred = _supervised.check_clusterings(labels_true, labels_pred)
    value = _supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)
# =============================================================================

X = sio.loadmat('Trento.mat')['HSI']
[m,n,l] = X.shape
X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
X = np.float32(X)

CD = sio.loadmat('Trento.mat')['Lidar']# CD = Complementary Data
CD = CD[:,:,0]
CD = CD.reshape((m,n,1))
[_,_,l_1] = CD.shape
CD = np.reshape(CD,(CD.shape[0]*CD.shape[1],l_1))
CD = min_max_scaler.fit_transform(CD)
CD = np.float32(CD)

y = sio.loadmat('Trento.mat')['GT']
y = np.reshape(y,(y.shape[0]*y.shape[1],-1))
y_test = y.reshape((m*n))
ind = np.nonzero(y)



# =============================================================================
# Number of Latent Features and Clusters
# =============================================================================
no_features =20
N_cluster = 6
# =============================================================================

# =============================================================================
# MDC main architechture
# =============================================================================


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(l, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, no_features),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(no_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, l),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        code = x
        x = self.decoder(x)
        return x, code



class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        # encoding layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=l_1,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, no_features, 5, 1, 2),
            nn.BatchNorm2d(no_features),
            nn.ReLU(),
        )

        # decoding layers
        self.dconv1 = nn.Sequential(
            nn.Conv2d(no_features, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.dconv2 = nn.Sequential(
            nn.Conv2d(128, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.dconv3 = nn.Sequential(
            nn.Conv2d(64, l_1, 5, 1, 2),
            nn.BatchNorm2d(l_1),
            nn.ReLU(),#Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        code = x
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = x.view(x.size(0), -1)
        code = code.view(code.size(1), -1)
        return x, code
    
class Fused_AE(nn.Module):
    def __init__(self):
        super(Fused_AE, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(no_features*2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, l+l_1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
# =============================================================================

# =============================================================================
# Setup
# =============================================================================

LR = 0.001
ae = AE()
ae.cuda()

cae = CAE()
cae.cuda()


fusion_cae_ae = Fused_AE()
fusion_cae_ae.cuda()

Spat_cof = 0.0001
Spec_cof = 0.0001
Fused_cof = 1

Iter = 500
thr = 100
print(ae)
print(cae)
print(fusion_cae_ae)


optimizer_ae = torch.optim.Adam(ae.parameters(), lr=LR)
optimizer_cae = torch.optim.Adam(cae.parameters(), lr=LR)
optimizer_fused = torch.optim.Adam(fusion_cae_ae.parameters(), lr=LR)
loss_func = nn.MSELoss()
#=============================================================================


# =============================================================================
# MDC Optimization Section
# =============================================================================
start_time = time.time()
Spectral_Data = torch.from_numpy(X)
tmpt_CD_org = CD.transpose()
tmpt_CD_loss = CD.reshape((1,m*n*l_1))
tmpt_CD = tmpt_CD_org.reshape((1,l_1,m,n))
Spatial_Data = torch.from_numpy(tmpt_CD)
Cat = np.concatenate([X,CD],1)
Cat = torch.from_numpy(Cat)
for i in range(Iter):
    Spec = Spectral_Data
    Spat = Spatial_Data
    output_ae, code_ae = ae(Spec.cuda())
    loss_ae = loss_func(output_ae, Spec.cuda())
    output_cae, code_cae = cae(Spat.cuda())
    loss_cae = loss_func(output_cae, torch.from_numpy(tmpt_CD_loss).cuda())
    code_cae = torch.transpose(code_cae, 0, 1)
    code = torch.cat([code_cae,code_ae],1)
    code.cuda()
    output_fuse = fusion_cae_ae(code)
    loss_fuse = loss_func(output_fuse, Cat.cuda())
    loss = (Fused_cof*loss_fuse) + (Spec_cof*loss_ae) + (Spat_cof*loss_cae)
    optimizer_ae.zero_grad()
    optimizer_cae.zero_grad()
    optimizer_fused.zero_grad()
    loss.backward()
    optimizer_ae.step()
    optimizer_cae.step()
    optimizer_fused.step()
    # loss_ls[1, i] = loss 
    print('Iteration: ', i, '| Total loss: %.4f' % loss.data.cpu().numpy())
    if loss.data.cpu().numpy() < thr:
        torch.save(ae.state_dict(), 'net_params_AEReconsTrento.pkl')
        torch.save(cae.state_dict(), 'net_params_CAEReconsTrento.pkl')
        torch.save(fusion_cae_ae.state_dict(), 'net_params_AE_CAEReconsTrento.pkl')
        thr = loss.data.cpu().numpy()
#=============================================================================



# =============================================================================
# Load Optimal Parameters (i.e., Weights and Bias)
# =============================================================================
ae1 = AE().cuda()
model_dict = ae1.state_dict()
pretrained_dict = torch.load('net_params_AEReconsTrento.pkl')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)  
ae1.load_state_dict(model_dict)


cae1 = CAE().cuda()
model_dict = cae1.state_dict()
pretrained_dict = torch.load('net_params_CAEReconsTrento.pkl')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)  
cae1.load_state_dict(model_dict)







Z_1 = ae1.encoder(Spectral_Data.cuda())
Z_1 = Z_1.detach().cpu().numpy()


Z = cae1.conv1(torch.from_numpy(tmpt_CD).cuda())
Z = cae1.conv2(Z)
Z = cae1.conv3(Z)
Z = Z.detach().cpu().numpy()
Z = Z.reshape((no_features,m*n))
Z = Z.transpose()


Z = np.concatenate([Z_1,Z],1)
# =============================================================================



# =============================================================================
# Clustering Section
# =============================================================================

# SC = SpectralClustering(n_clusters=N_cluster, assign_labels="discretize", random_state=0, affinity='nearest_neighbors')
# CS = SC.fit(Z)
KM = KMeans(n_clusters=N_cluster, random_state=0)
CS = KM.fit(Z)
CSmap = np.zeros((m*n))
CSmap = CS.labels_ + 1
CA = clustering_accuracy(y_test[ind[0]], CSmap[ind[0]])
NMI = normalized_mutual_info_score(y_test[ind[0]], CSmap[ind[0]])
ARI = adjusted_rand_score(y_test[ind[0]], CSmap[ind[0]])
print('CA:\t'+np.str(CA)+'\n'+'NMI:\t'+np.str(NMI)+'\n'+'ARI:\t'+np.str(ARI))
CSmap = CSmap.reshape((m,n))
    



# =============================================================================
# Visualization
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('Clustering result via MDC')
ax1.imshow(y_test.reshape((m,n)))
ax1.set_title('GT')
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax2.imshow(CSmap)
ax2.set_title('MDC')
ax2.set_yticklabels([])
ax2.set_xticklabels([])
end_time = time.time()


P_time = end_time - start_time
print(P_time)


# =============================================================================  
    
# =============================================================================
# Saving the Clustering Map as .mat file 
# =============================================================================
sio.savemat('CSmap_MDC.mat', {'CSmap':CSmap})    
# ============================================================================= 
    
    
    
