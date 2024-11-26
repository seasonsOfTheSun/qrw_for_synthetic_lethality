
import networkx as nx 
import numpy as np
import pandas as pd
import itertools as it
import scipy.sparse
import json

from eigenvectors import calculate_evecs

def main():
    
    folder = "data/intermediate/fourth_large"
    
    # here we create a simulation that models some features
    # of real disease etitology.
    
    # we consider an arbitrary network genes, 
    # and some number of gene modules of constant size
    # n_modules,module_size,total_genes = 10,10,200
    n_modules,module_size,total_genes = 30,10,1000
    gt = GroundTruth(n_modules,module_size,total_genes)
    gene_list = list(gt.genes)
    
    
    # determines the characteristics of the 
    # simulated PPI, specifically the probability
    # of connection between nodes in a single gene module
    # and 
    # Creates a network on the genes, where the modules are especially dense and
    # strongly connected.
    p_inter,p_intra = 0.02,1.0
    gt.ppi = PPI(gt,
                p_inter,p_intra)
    
    pos = nx.spring_layout(gt.ppi.network)


    # we select two of the gene modules to be the disease modules 
    # in each 'patient' these genes can be 'mutated' or not 'mutated'
    # when BOTH two disease modules have more mutations than some threshold (default 1)
    # we consider the 'patient' to have the disease
    gt.generate_disease_combinatorics()

    # here we simluate a population of patients who may or may not have the disease
    # by sampling uniformly over the genes with patient probab
    # ranfomly 'mutating' some of the genes and using the ground truth combinatorics
    # to determine if they have the disease or not

    mutation_p = 0.2
    n_patients = 200
    
    all_patient_mutations = []
    all_disease_statuses = []

    for i in range(n_patients):
        patient_mutations = gt.simulate_mutations(mutation_p)
        patient_disease_status = gt.check_disease_status(patient_mutations)
        
        all_patient_mutations.append(patient_mutations)
        all_disease_statuses.append(patient_disease_status)


    # THIS IS THE POINT OF THE WHOLE MODULE!
    # here we create an object, based on the patient data
    # that calculates the behaviour 
    # of a matrix that can hopefully be modified into 
    # and interaction Hamiltonian.
    H = DataInteractionMatrix(gene_list,
                              all_patient_mutations,
                              all_disease_statuses
                              )
    

    # Calculates the Symmetric-Normalized Adjacency Eigenvectors
    n_evecs = 10
    adjacency = nx.adjacency_matrix(gt.ppi.network,nodelist=gt.ppi.nodelist)
    evalues,evecs = calculate_evecs(adjacency,n_evecs
                                    )

    
    import os
    os.makedirs(folder,exist_ok=True)
    
    df = pd.DataFrame(evecs,columns=[f"Eigenvector-{i}" for i in range(n_evecs)],
                 index=gene_list)
    df = df/np.linalg.norm(evecs,axis=0)
    df.to_csv(f"{folder}/evecs.csv")
    
    # The inner products between the modules in the constructed matrix, 
    # showing how the two disease modules (tensored together) have a much higher 
    # value in <(module tensor module) | M |(module tensor module)> than the others
    out=np.zeros((n_modules,n_modules))
    for i,j in it.product(range(n_modules),repeat=2):
        module_1,module_2 = gt.modules[i],gt.modules[j]
        v = np.array([int(i in module_1) for i in gene_list])
        w = np.array([int(i in module_2) for i in gene_list])
        out[i,j] = H.bilinear_form_magnitude(v,w)
        
    df = pd.DataFrame(out,columns=[f"Module-{i}" for i in range(n_modules)],index=[f"Module-{i}" for i in range(n_modules)]) 
    df.to_csv(f"{folder}/module_inner_products.csv")   
    

    out=np.zeros((n_evecs,n_evecs))
    for i,j in it.product(range(n_evecs),repeat=2):

        v=evecs[:,i]
        w=evecs[:,j]

        out[i,j] = H.bilinear_form_magnitude(v,w)
    
    df = pd.DataFrame(out,columns=[f"Eigenvector-{i}" for i in range(n_evecs)],index=[f"Eigenvector-{i}" for i in range(n_evecs)]) 
    df.to_csv(f"{folder}/eigenvector_inner_products.csv")   

    # The inner products between the PPI evectors in the constructed matrix, 
    # showing how the two PPI evectors representing the modules (tensored together) have a much higher 
    # value in <(evec tensor evec) | M |(evec tensor evec)> than the others
    out = []
    for i in range(n_modules):
        module = gt.modules[i]
        v = np.array([int(i in module) for i in gene_list])
        out.append(evecs.T @ v)
    
    df = pd.DataFrame(out,columns=[f"Eigenvector-{i}" for i in range(n_evecs)],index=[f"Module-{i}" for i in range(n_modules)]).T
    df.to_csv(f"{folder}/eigenvector_module_loadings.csv")   
    
    parameters = {"n_modules":n_modules,
                  "module_size":module_size,
                  "total_genes":total_genes,
                  "p_inter":p_inter,
                  "p_intra":p_intra,
                  "n_evecs":n_evecs,
                  "mutation_p":mutation_p,
                  "n_patients":n_patients}
    
    
    nx.write_gml(gt.ppi.network,f"{folder}/network.gml")

    json.dump({i:list(v) for i,v in pos.items()}, open(f"{folder}/pos.json", 'w'))

    json.dump(parameters, open(f"{folder}/parameters.json", 'w'))
    json.dump(gt.ppi.nodelist, open(f"{folder}/nodelist.json", 'w'))
    json.dump(gt.modules, open(f"{folder}/modules.json", 'w'))
    json.dump(gt.disease_modules, open(f"{folder}/disease_modules.json", 'w'))
    json.dump(gt.disease_module_indices, open(f"{folder}/disease_module_indices.json", 'w'))
    json.dump([[str(v) for v in w] for w in all_patient_mutations], open(f"{folder}/all_patient_mutations.json", 'w'))
    json.dump(all_disease_statuses, open(f"{folder}/all_disease_statuses.json", 'w'))



    
class PatientInteractionMatrix:
    
    def __init__(self,nodelist,mutations):
        """ 
        Creates a matrix of the form
        sum_{ij} (|i> \tensor |j>)(<i| \tensor <j|)
        where i,j are all pairs of genes in a particual patient
        
        """

        self.nodelist = nodelist
        self.mutations=mutations

        node_to_index = {node:index for index,node in enumerate(nodelist)}
        self.gene_pair_indices = [(node_to_index[gene1],node_to_index[gene2])
                                  for gene1,gene2 in it.product(mutations,repeat=2)]
        
    def bilinear_form_magnitude(self,v,w):
        """ Applies the patient matrix as bilinear form,
        calculating (<v| \ tensor <w|) A (|v> \ tensor |w>)
        where A is the patient-specific interaction matrix
        held by this PatientInteractionMatrix object """
        out = 0
        for i,j in self.gene_pair_indices:
            out += (v[i]*w[j])**2
        return out
    
    def as_scipy_sparse_matrix(self):
        
        n = len(self.nodelist)
        out = scipy.sparse.zeros((n**2,n**2))
        
        for i,j in self.gene_pair_indices:
            out[n*i+j,n*i+j]=1
        
        return out
        
        
class DataInteractionMatrix:
    
    def __init__(self,nodelist,mutations,disease_statuses,patient_weighting_map=None):
        """ 
        Creates a matr}ix of the form
        sum_{patients} weight{patient} sum_{ij} (|i> \ tensor |j>)(<i| \ tensor <j|)
        where i,j are all pairs of genes in a particual patient,
        summed over all the patients selected, and a weight is 
        given according to whether the patirnt has the disease or not.
        """
        self.nodelist = nodelist
        
        if patient_weighting_map == None:
            patient_weighting_map = {}
            patient_weighting_map[True] =  1.0
            patient_weighting_map[False]  = -1.0
        
        self.patient_interaction_matrices = []
        self.patient_weightings = []
        for patient_mutations,ds in zip(mutations,disease_statuses):
             
             tmp = PatientInteractionMatrix(nodelist,patient_mutations)
             self.patient_interaction_matrices.append(tmp)
             self.patient_weightings.append(patient_weighting_map[ds])

             
        
    def bilinear_form_magnitude(self,v,w):
        """ Applies the data matrix as bilinear form (<v| \ tensor <w|) A (|v> \ tensor |w>)  """
        out = 0
        for pim,weight in zip(self.patient_interaction_matrices,
                         self.patient_weightings):
            out += weight*pim.bilinear_form_magnitude(v,w)
        return out
            
    def as_scipy_sparse_matrix(self):
        
        n = len(self.nodelist)
        out = scipy.sparse.zeros((n**2,n**2))
        
        for pim,weight in zip(self.patient_interaction_matrices,
                        self.patient_weightings):
            out += weight*pim.as_scipy_sparse_matrix()
        return out       
    

            

               

class GroundTruth:
    
    def __init__(self,module_size,n_modules,total_nodes):
        
        assert module_size*n_modules < total_nodes

        self.module_size = module_size 
        self.n_modules = n_modules
        self.total_nodes = total_nodes
        
        self.modules  = [list(range(module_size*n,module_size*(n+1))) for n in range(n_modules)]
        self.genes = list(range(total_nodes))
        
        
    def generate_disease_combinatorics(self,threshold=1):
        """ 
        Randomly generates a choice of ground truth for the disease etiology model.
        Randomly selecting a
        pair of gene modules, henceforth the disease modules
        and a threshold number of mutations that suffice for those 
        modules to be considered inactivated. 
        """
        
        i = np.random.choice(self.n_modules*(self.n_modules-1)//2)
        n,m = list(it.combinations(range(self.n_modules),2))[i]
        self.disease_module_indices = (n,m)
        self.disease_modules = [self.modules[n],self.modules[m]]
        
        self.thresholds = [threshold,threshold]
        
        self.interaction_type = "lethality"

        
    def simulate_mutations(self,p):
        
        n = int(p*len(self.genes))
        return np.random.choice(self.genes,size=(n,))
          
        
    def check_disease_status(self,mutations):
        """ For the ground truth model held in this object,
            would a given a set of mutations give rise to the disease?
            
            If interaction_type is "lethality", it checks that
            both of the disease modules specified by the current ground truth 
            contain at least 'threshold' number of mutations, and are therefore 
            inactivated.
            
            If interaction_type is "rescue", it checks that one or the other disease module, 
            but NOT both, contain at least 'threshold' number of mutations.
            
            note that by default 'threshold' is 1
            
            

        Args:
            mutations (list): a list of genes (nodes in the PPI network)
            that describes the patients mutations.

        Returns:
            bool: according to the current disease model
        """
        
        if self.interaction_type == "lethality":
            out = True
            for module,threshold in zip(self.disease_modules, self.thresholds):
                n_mutated_in_module = len(set(module)&set(mutations))
                out &= (n_mutated_in_module >= threshold)
            return out
        
        elif self.interaction_type == "rescue":
            
            assert len(self.disease_modules) == 2
            
            out = False
            for module,threshold in zip(self.disease_modules, self.thresholds):
                n_mutated_in_module = len(set(module))
                if (n_mutated_in_module >= threshold):
                    out = (not out)
                    
            return out
                

        
        

            
        
            
        

class PPI:
    
    def __init__(self,gt,
                 p_inter,p_intra):
        """ Helper class for generating simulated PPI Networks

        Args:
            gt (GroundTruth): An object that holds info about the disease modules and total genes in the model
            p_inter (float, 0 to 1): the probability of connection between genes in different disease modules or not in disease modules
            p_intra (float, 0 to 1): the probability of connection between genes in a disease module
        """
        
        assert p_inter < p_intra
        
        
        self.network = nx.erdos_renyi_graph(gt.total_nodes,p_inter)
        
        
        for module in gt.modules:
            for i,j in it.combinations(module,2):
                if np.random.rand() < p_intra-p_inter:
                    self.network.add_edge(i,j)
        
        self.nodelist = list(self.network.nodes())
        
        

main()