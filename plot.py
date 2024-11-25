import matplotlib.pyplot as plt
import networkx as nx 
import numpy as np
import pandas as pd
import json
import os
import seaborn as sns

name = "fourth_large"
folder = f"data/intermediate/{name}"
figfolder = f"figures/{name}"


true_module_indices = json.load(open(f"{folder}/disease_module_indices.json"))
print(f"The true disease modules are {true_module_indices}")
os.makedirs(figfolder,exist_ok=True)

ppi = nx.read_gml(f"{folder}/network.gml")
pos = json.load(open(f"{folder}/pos.json"))

fig,ax = plt.subplots()
nx.draw_networkx_nodes(ppi,pos=pos,ax=ax,node_size=10,node_color="w",edgecolors="k",linewidths=0.3)
nx.draw_networkx_edges(ppi,pos=pos,ax=ax,width=0.3)
fig.savefig(f"{figfolder}/network.png",dpi=300,bbox_inches='tight')
fig.savefig(f"{figfolder}/network.pdf")



modules = json.load(open(f"{folder}/modules.json"))
nodelist = json.load(open(f"{folder}/nodelist.json"))
     
     
os.makedirs(f"{figfolder}/network_modules",exist_ok=True)
for module_index in range(len(modules)):
    node_color = [i in modules[module_index] for i in nodelist]
    fig,ax = plt.subplots()
    nx.draw_networkx_nodes(ppi,pos=pos,ax=ax,node_color=node_color,edgecolors="k",cmap="binary",node_size=10,linewidths=0.3)
    nx.draw_networkx_edges(ppi,pos=pos,ax=ax,width=0.3)
    fig.suptitle(f"module {module_index}")
    
    fig.savefig(f"{figfolder}/network_modules/module_{module_index}.png",dpi=300,bbox_inches='tight')
    fig.savefig(f"{figfolder}/network_modules/module_{module_index}.pdf")

os.makedirs(f"{figfolder}/network_evecs",exist_ok=True)
evecs = pd.read_csv(f"{folder}/evecs.csv",index_col=0)

for evec_index in evecs.columns:
    node_color = evecs.loc[:,evec_index]
    vmax = np.max(np.abs(node_color))
    fig,ax = plt.subplots()
    nx.draw_networkx_nodes(ppi,pos=pos,ax=ax,node_color=node_color,edgecolors="k",cmap="seismic",node_size=10,vmin=-vmax,vmax=vmax,linewidths=0.3)
    nx.draw_networkx_edges(ppi,pos=pos,ax=ax,width=0.3)

    fig.suptitle(f"Eigenvector {evec_index}")
    fig.savefig(f"{figfolder}/network_evecs/evecs_{evec_index}.png",dpi=300,bbox_inches='tight')
    fig.savefig(f"{figfolder}/network_evecs/evecs_{evec_index}.pdf")


df = pd.read_csv(f"{folder}/eigenvector_module_loadings.csv",index_col=0)
print(df)
fig,ax= plt.subplots()
vmax = np.abs(df).max().max()
cbar_label = r"$(<\mathrm{module}_i|\mathrm{evec}_j>$"
sns.heatmap(df,ax=ax,cmap="seismic",vmin=-vmax,vmax=vmax,cbar_kws={"label":cbar_label})
fig.suptitle(f"The true disease modules are {true_module_indices}")
ax.axis("tight")
fig.savefig(f"{figfolder}/eigenvector_loadings.png",dpi=300,bbox_inches='tight')




df= pd.read_csv(f"{folder}/module_inner_products.csv",index_col=0) 

fig,ax= plt.subplots()
vmax = np.abs(df).max().max()
cbar_label = r"$(<\mathrm{module}_i| \otimes <\mathrm{module}_j|) H (|\mathrm{module}_i> \otimes |\mathrm{module}_j>)$"
sns.heatmap(df,ax=ax,cmap="seismic",cbar_kws={"label":cbar_label})#,vmin=-vmax,vmax=vmax)
ax.axis("tight")
fig.suptitle(f"The true disease modules are {true_module_indices}")
fig.savefig(f"{figfolder}/module_inner_products.png",dpi=300,bbox_inches='tight')


    
df= pd.read_csv(f"{folder}/eigenvector_inner_products.csv",index_col=0) 
fig,ax= plt.subplots()
vmax = np.abs(df).max().max()
cbar_label = r"$(<\mathrm{evec}_i| \otimes <\mathrm{evec}_j|) H (|\mathrm{evec}_i> \otimes |\mathrm{evec}_j>)$"
sns.heatmap(df,ax=ax,cmap="seismic",cbar_kws={"label":cbar_label})#,vmin=-vmax,vmax=vmax)
ax.axis("tight")
fig.savefig(f"{figfolder}/eigenvector_inner_products.png",dpi=300,bbox_inches='tight')


print(f"figures written to {figfolder}")

"""
    json.dump(parameters, open(f"{folder}/parameters.json", 'w'))
    json.dump(gt.ppi.nodelist, open(f"{folder}/nodelist.json", 'w'))
    json.dump(gt.modules, open(f"{folder}/modules.json", 'w'))
    json.dump(gt.disease_modules, open(f"{folder}/disease_modules.json", 'w'))
    json.dump(gt.disease_module_indices, open(f"{folder}/disease_module_indices.json", 'w'))
    json.dump([[str(v) for v in w] for w in all_patient_mutations], open(f"{folder}/all_patient_mutations.json", 'w'))
    json.dump(all_disease_statuses, open(f"{folder}/all_disease_statuses.json", 'w'))


    df.to_csv(f"{folder}/evecs.csv")
    



    df.to_csv(f"{folder}/module_inner_products.csv")   
    


    
    df.to_csv(f"{folder}/eigenvector_inner_products.csv")   

    
    df.to_csv(f"{folder}/eigenvector_module_loadings.csv")   
""" 