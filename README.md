
# Model of Disease Etiology
Here we create a toy model that simulates some features of genetic disease etiology,in partiular the synthetic lethality of mutations.
    

### Modules and Disease Definition

We consider a set of **genes**, some of which are grouped into disjoint (non-overlapping) **gene modules**, two of which are arbitrarily chosen to be the ground truth **disease modules**.  Each disease module which has a specified **threshold** number of mutations, beyond which it is considered to be **inactivated**. If for a given set of mutated genes, *both* disease modules are inactivated, then the set of mutations causes **disease**


```python
n_modules,module_size,total_genes = 30,10,1000
gt = GroundTruth(n_modules,module_size,total_genes)
gene_list = list(gt.genes)
gt.generate_disease_combinatorics()
 ```

### PPI / Gene Similarity Network

We randomly generate a network with the genes as nodes, which we call the **PPI**. We want to capture the idea that, like in the real PPI, genes of similar function are connected by a link. So we specify a probability $p_{\mathrm{intra}}$ of genes within a module having a link between them, and a probability $p_{\mathrm{inter}}$ of genes not in the same module having a link between them, and generate the links with independent Bernoulli trials (coin flips). 

Note that not every gene in the network is necessarily in a module, and the disease modules are not treated in any way differently then the non-disease gene modules.

```python
p_inter,p_intra = 0.02,1.0
gt.ppi = PPI(gt,p_inter,p_intra)
```

### Patient Dataset

Given this specified ground truth, we simulate a population of **patients** who may or may not have the disease, by sampling a selection of the genes for each patient, and considering the selected set to be the set of **mutations** for that patient.

Using the previously specified ground truth, we determine if each of the patients **has the disease or not**. This gives us a table of mutations and disease statuses, replicating the kind of data we have in order to discover causal mutations  in real life.

```python
mutation_p = 0.2
n_patients = 200
    
all_patient_mutations = []
all_disease_statuses = []

for i in range(n_patients):
        patient_mutations = gt.simulate_mutations(mutation_p)
        patient_disease_status = gt.check_disease_status(patient_mutations)
        
    all_patient_mutations.append(patient_mutations)
    all_disease_statuses.append(patient_disease_status)

```
# Recovering the Ground Truth from Simulated Patient Data

Given access only to this simulated patient data in the form of mutations and corresponding disease status, we would like to see if we can recover the ground truth disease modules. To this end we use it  a matrix on the space $V \otimes V$ that can 'detect' the relevant disease modules in some sense specified further below. The matrix is given by

$$ \sum_{k} w_k \sum_{i \neq j \in M_k} (|i\rangle \otimes | j \rangle) (\langle i|\otimes \langle j |)  $$

Where 
* $k$ is the patient, 
* $M_k$ is the patient $k$'s specific set of mutations, and
* $w_k$ is a weighting that depends on the patient's disease status
* $|i\rangle$ is the element of $V$ in which the node $i$ is given value $1$ and all others $0$.






```python
H = DataInteractionMatrix(
                        gene_list,
                        all_patient_mutations,
                        all_disease_statuses
                        )
```  


## From the Gene Modules

We first check that we can recover the chosen disease modules from the full list of gene modules. 

This is substantially easier than the problem faced in reality, and fact itreduces to a simple (albeit potentially intractible) combinatorial search. 

However, we can use this case to test the utility of the previously constructed matrix $M$ can be shown by the fact that, at least for some parameters of the network and ground truth,

$$ (\langle G_i|\otimes \langle G_j |)
 | M |  (|G_i\rangle \otimes | G_j \rangle)
$$

is largest by an overwhelming margin when $G_i$ and $G_j$ are the true disease modules, amongst all $G_i \neq G_j$.


```python

out=np.zeros((n_modules,n_modules))
for i,j in it.product(range(n_modules),repeat=2):
    module_1,module_2 = gt.modules[i],gt.modules[j]
    v = np.array([int(i in module_1) for i in gene_list])
    w = np.array([int(i in module_2) for i in gene_list])
    out[i,j] = H.bilinear_form_magnitude(v,w)
    
module_inner_products = pd.DataFrame(out,
columns=[f"Module-{i}" for i in range(n_modules)],#
index=[f"Module-{i}" for i in range(n_modules)]) 

```



## From the PPI

However, in reality, we don't actually have access to the the true gene modules, let alone know
which of them are the disease gene modules. However, we have the PPI, and the guiding principle that gene modules frequently form dense clusters within the PPI. Connected clusters of nodes in a network are often highlighted 
as high-eigenvalue eigenvectors of the adjacency matrix. 


So with this in mind, we take we calculate the Symmetric-Normalized Adjacency Eigenvectors for the PPI network. We choose the Symmetric-Normalized Adjacency because this is the adjacency matrix that forms the principal terms of the Hamiltonian of the QRW.

$$ A\otimes I + I \otimes A + H_{\mathrm{int}}$$

Hence it makes sense to construct the behaviour of the putative interaction Hamiltonian $H_{\mathrm{int}}$ to have specific effects on the vectors  of the form |$v_\lambda \rangle \otimes | v_\mu \rangle$, as these will already have large eigenvector $\lambda + \mu$, a prominent role in the long term beahvious of the random walk, and the role of $H_{\mathrm{int}}$ will be to adjust their relative prominence. 





```python

n_evecs = 10
adjacency = nx.adjacency_matrix(gt.ppi.network,nodelist=gt.ppi.nodelist)
evalues,evecs = calculate_evecs(adjacency,n_evecs
                                    )
evecs = pd.DataFrame(evecs,
    columns=[f"Eigenvector-{i}" for i in range(n_evecs)],
                 index=gene_list)
evecs = evecs/np.linalg.norm(evecs,axis=0)

out = []
for i in range(n_modules):
    module = gt.modules[i]
    v = np.array([int(i in module) for i in gene_list])
        out.append(evecs.T @ v)
    
    eigenvector_module_loadings = pd.DataFrame(out,
                    columns=[f"Eigenvector-{i}" for i in range(n_evecs)],index=[f"Module-{i}" for i in range(n_modules)]).T
```



As in the case with the gene modules,  the matrix $M$ can be shown (at leasrt for some parameters of the PPI and ground truth construction, and when the PPI eigenvectors contain good representatives of the modules) to have the property that 

$$ (\langle v_\lambda |\otimes \langle  v_\mu |)
 | M |  (|v_\lambda \rangle \otimes | v_\mu \rangle)
$$

is largest by an overwhelming margin when $v_\lambda$ and $v_\mu$ are the eigenvectors which correlate most strongly with the true disease modules, amongst all $v_\lambda \neq v_\mu$.






```python

out=np.zeros((n_evecs,n_evecs))
for i,j in it.product(range(n_evecs),repeat=2):

    v=evecs[:,i]
    w=evecs[:,j]

    out[i,j] = H.bilinear_form_magnitude(v,w)
    
    eigenvector_inner_products = pd.DataFrame(out,columns=[f"Eigenvector-{i}" for i in range(n_evecs)],index=[f"Eigenvector-{i}" for i in range(n_evecs)]) 
```



