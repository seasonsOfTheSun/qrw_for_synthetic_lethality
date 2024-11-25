
import scipy.sparse
import scipy.stats
import numpy as np
import pandas as pd


def calculate_evecs(G, n_evectors):
  # make the symmetric normalized adjacency
  # whose eigenvectors are the spectral eigennvectors
  # N.B the normalized adjacency is I - normalized Laplacian
  # with

  n_nodes = G.shape[0]
  d = G @ np.ones(n_nodes)
  D = scipy.sparse.diags(d)
  D_neg_half = scipy.sparse.diags(d**(-1/2))
  normalized_adjacency = D_neg_half @ G @ D_neg_half

  # calculate the eigenvectors for the normnalized adjacenncy
  # which are also the eigenvectors for the normnalized Laplacian
  # but by doing it this way round we cut-off the ones we don't want
  e,evecs = scipy.sparse.linalg.eigsh(normalized_adjacency, k = n_evectors)

  # `scipy.sparse.linalg.eigsh` returns the eigenvectors and eigenvalues in
  # increasing eigenvalue order

  # we reorder the eigenvalues and eigenvectors by decreasing eigenvalue,
  # since high adjacency-eigenvallue is low Laplacian eigennvalue
  e = e[::-1]
  evecs = evecs[:,::-1]

  return e,evecs
