from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans, OPTICS

def kmeans_(n):
    return KMeans(n_clusters=n, random_state=0)

def optics_(n):
    return OPTICS(min_samples=n, metric='nan_euclidean')


def compute_number_clusters(data, model, metric, min_=False):
    '''
        min_: if True the evaluated metric is a minimization problem 
    '''
    if min_:
        best_score = 100000
    else:
        best_score = 0
    best_n = 0
    n = 2
    while 1:
        cluster = model(n).fit(data)
        labels = cluster.labels_
        score = metric(data, labels)
        print((score, n))
        if min_:
            if score < best_score:
                best_score = score
                best_n = n
        else:
            if score > best_score:
                best_score = score
                best_n = n
        n += 1
        if n > 10:
            break
    return best_score, best_n