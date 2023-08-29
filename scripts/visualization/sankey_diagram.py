import numpy as np
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from webcolors import hex_to_rgb

def create_flow(data1, data2, label1, label2, Y):
    for i in np.unique(data1):
        ind_0 = np.where(data1==i)[0].tolist()
        for j in np.unique(data2):
            ind_1 = np.where(data2==j)[0].tolist()
            intersection = list(set(ind_0).intersection(set(ind_1)))
            pass_ = np.where(Y[intersection]==0)[0].tolist()
            fail = np.where(Y[intersection]==1)[0].tolist()
            source.append(label1+'_'+str(i))
            target.append(label2+'_'+str(j))
            values.append(len(pass_))
            source.append(label1+'_'+str(i))
            target.append(label2+'_'+str(j))
            values.append(len(fail))

def create_sankey_diagram(labels_cluster_path, filename, percentile_list, Y, gamma=False, g=0):
    labels = {}
    for p in percentile_list:
        if gamma:
            labels[p] = np.loadtxt(labels_cluster_path+filename+str(p)+'_gamma_'+str(g)+'.txt', dtype=int)
        else:
            labels[p] = np.loadtxt(labels_cluster_path+filename+str(p)+'.txt', dtype=int)
   
    node_label = []
    for p in percentile_list:
        node_label.extend(["Week"+str(p).split('.')[1]+'_'+str(i) for i in np.unique(labels[p])])
       
    node_dict = {y:x for x, y in enumerate(node_label)}
    
    global source 
    source = []
    global target 
    target = []
    global values 
    values = []

    for i in range(len(percentile_list)-1):
        p = percentile_list[i]
        next_p = percentile_list[i+1]
        create_flow(labels[p], labels[next_p], "Week"+str(p).split('.')[1], "Week"+str(next_p).split('.')[1], Y)

    source_node = [node_dict[x] for x in source]
    target_node = [node_dict[x] for x in target]

    node_color = ['#00A36C', '#FF6347']
    link_color = [node_color[x%2] for x in range(len(source))]

    link_color = ['rgba({},{},{}, 0.4)'.format(
        hex_to_rgb(x)[0],
        hex_to_rgb(x)[1],
        hex_to_rgb(x)[2]) for x in link_color] 

    fig = go.Figure( 
        data=[go.Sankey( # The plot we are interest
            # This part is for the node information
            node = dict( 
                label = node_label,
                color = ['#D3D3D3' for _ in node_label]
            ),
            # This part is for the link information
            link = dict(
                source = source_node,
                target = target_node,
                value = values,
                color = link_color,
            ))])

    # With this save the plots 
    plot(fig,
        image_filename='sankey_plot_1', 
        image='png', 
        image_width=1000, 
        image_height=600
    )
    # And shows the plot
    fig.show()