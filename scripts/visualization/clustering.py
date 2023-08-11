from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_tsne_and_pca(data, labels, pca_before=False):    
    #PCA
    pca2 = PCA(n_components=2)
    PCA2 = pca2.fit_transform(data)
    dfpca = pd.DataFrame(PCA2)
    dfpca['cluster'] = labels
    dfpca.columns = ['x1','x2','cluster']
    
    # TSNE
    if pca_before:
        pca7 = PCA(n_components=7)
        PCA7 = pca7.fit_transform(data) 
    
        Xtsne = TSNE(n_components=2).fit_transform(PCA7)
        dftsne = pd.DataFrame(Xtsne)
        dftsne['cluster'] = labels
        dftsne.columns = ['x1','x2','cluster']
    else:
        Xtsne = TSNE(n_components=2).fit_transform(data)
        dftsne = pd.DataFrame(Xtsne)
        dftsne['cluster'] = labels
        dftsne.columns = ['x1','x2','cluster']
        
    # UMAP
    reducer = umap.UMAP(n_components=2)
    umap2 = reducer.fit_transform(data)
    dfumap = pd.DataFrame(umap2)
    dfumap['cluster'] = labels
    dfumap.columns = ['x1', 'x2', 'cluster']
    
    fig, ax = plt.subplots(1, 3, figsize=(18,6))
    sns.scatterplot(data=dftsne,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,ax=ax[0])
    ax[0].set_title('Visualized on TSNE 2D')
    sns.scatterplot(data=dfpca,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,ax=ax[1])
    ax[1].set_title('Visualized on PCA 2D')
    sns.scatterplot(data=dfumap,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,ax=ax[2])
    ax[2].set_title('Visualized on UMAP 2D')
    plt.show()