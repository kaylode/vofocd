import time
import numpy as np
from numpy.lib.npyio import load
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser('Plot PCA or t-SNE')
parser.add_argument('--numpy_dir', type=str, help='Directory contains all numpy features.')
parser.add_argument('--label_csv', type=str, help='CSV file contain numpy labels.')
parser.add_argument('--algorithm', default='pca', type=str, help='Dimensionality reduction algorithm')

def pca_reduce(data, num_components=100):
    pca = PCA(n_components=num_components)
    result = pca.fit_transform(data)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    return result

def tsne_reduce(data, num_components=100, perplexity=40, n_iter=300, verbose=False):
    tsne = TSNE(n_components=num_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(data)
    return tsne_results

def dim_reduce(data, num_components=100, reduction_algorithm='pca'):
    if reduction_algorithm == 'pca':
       result = pca_reduce(data, num_components=num_components)

    elif reduction_algorithm == 'tsne':
        result = tsne_reduce(data, num_components=num_components)

    else:
        raise ValueError('Algorithm is not supported')

    return result

def plot2D(df, num_classes=10):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="dim1", y="dim2",
        hue="y",
        palette=sns.color_palette("hls", num_classes),
        data=df,
        legend="full",
        alpha=0.3
    )

def plot3D(df):
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=df.loc[rndperm,:]["dim1"], 
        ys=df.loc[rndperm,:]["dim2"], 
        zs=df.loc[rndperm,:]["dim3"], 
        c=df.loc[rndperm,:]["y"], 
        cmap='tab10'
    )
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')
    ax.set_zlabel('dim3')
    plt.show()

def load_numpy(npy_path):
    npy_feat = np.load(npy_path, mmap_mode='r')['feat']
    return npy_feat

def make_df(df_list):
    stacked_feats = []
    labels = []
    for filename, label in df_list:
        feat = load_numpy(filename)
        stacked_feats.append(feat)
        labels.append(label)

    stacked_feats = np.stack(stacked_feats, axis = 0)

    colnames = ['dim'+str(i) for i in range(stacked_feats.shape[1])]
    data_df = pd.DataFrame(stacked_feats, columns=colnames)
    data_df['y'] = labels

    return data_df

if __name__ == '__main__':
    args = parser.parse_args()

    label_df = pd.read_csv(args.label_csv)
    
    df_list = [(filename, label) for filename, label in zip(label_df.name, label_df.label)]

    df = make_df(df_list)
    










