import matplotlib.pyplot as plt
import numpy as np

def plot_concepts(Y, P, pat_name, pat_id, chars, labels):
    '''
    Args:
        pat_name: (str) Student pattern name
        pat_id: (int) Pattern index
        chars: (str) Student pattern characteristics (eg. high, low)
        labels: (list) Cluster labels
    '''
    # Get pattern student labels
    pat = P[:, pat_id]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for out, target in enumerate(['Pass', 'Fail']):
        students = []
        for c in range(np.max(np.unique(labels))+1):
            cluster = np.where(labels == c)[0]
            char_s = []
            for i, pat_char in enumerate(chars):
                # Get indices of specified characteristic
                char_id = i+1
                char_idx = np.argwhere(pat == char_id).flatten()
                a = list(set(char_idx) & set(list(cluster)))
                char_s.append((Y[a] == out).sum()/len(cluster))

            students.append(("Cluster "+str(c), char_s))
        
        ax[out].bar([i[0] for i in students], [i[1][0] for i in students], label=chars[0], color='tomato')
        ax[out].bar([i[0] for i in students], [i[1][1] for i in students], bottom=[i[1][0] for i in students], label=chars[1], color='c')
        if len(chars) > 2:
            ax[out].bar([i[0] for i in students], [i[1][2] for i in students], bottom=np.add([i[1][0] for i in students], [i[1][1] for i in students]), label=chars[2], color='mediumseagreen')
        ax[out].set_title(pat_name+' ('+target+')', fontsize=18)
        ax[out].set_ylabel('Percentage of students', fontsize=14)
        ax[out].set_xticklabels([i[0] for i in students], rotation=45, ha='right')
        ax[out].legend(fontsize=12)
    plt.show()
