from time import time
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
import torchvision.datasets as dsets
from mpl_toolkits.mplot3d import Axes3D
import os

from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn import decomposition
from torchvision import transforms
import torch

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X_emb, label,title=None):
    x_min, x_max = np.min(X_emb, 0), np.max(X_emb, 0)
    X_emb = (X_emb - x_min) / (x_max - x_min)

    plt.figure()

    for i in range(X_emb.shape[0]):
        plt.scatter(X_emb[i, 0], X_emb[i, 1]
                    , color=plt.cm.Set1(label[i] / 10.), cmap=plt.cm.Spectral(10))

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def sne(model, classifier, data):
    transform0 = transforms.Compose([
        # 输入图片缩放到32*32
        transforms.Scale(28),
        # 将图片变为【0，1】
        transforms.ToTensor()])
    mnist = dsets.MNIST('./media/lxh/document/demo-2020/3-UdisGAN-MU-EGD/data/mnist/MNIST/raw', train='train',
                        download=True, transform=transform0)
    mnist_loader = torch.utils.data.DataLoader(mnist,
                                               batch_size=1000)

    mnist_iter = iter(data)
    mnist_im, m_labels = mnist_iter.next()
    mnist_im, m_labels = mnist_im.cuda(), m_labels.cuda().long().squeeze()
    m = classifier(model(mnist_im))

    sub_sample = 1000
    y = m_labels[0:sub_sample]
    X = m[0:sub_sample]
    X = X.view(sub_sample, -1)
    n_samples, n_features = X.shape[0], X.shape[1]

    n_neighbors = 30

    # ----------------------------------------------------------------------
    # Plot images of the digits

    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    t0 = time()
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(X.cpu())

    plot_embedding(result, y.cpu(),
                   "t-SNE embedding of the digits (time %.2fs)" %
                   (time() - t0))

    plt.show()


