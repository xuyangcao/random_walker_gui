"""
Random walker algorithm

from *Random walks for image segmentation*, Leo Grady, IEEE Trans 
Pattern Anal Mach Intell. 2006 Nov;28(11):1768-83.

Dependencies:

* numpy >= 1.4, scipy

* optional: pyamg, numexpr

Installing pyamg and using the 'cg_mg' mode of random_walker improves
significantly the performance.

Installing numexpr makes only a slight improvement. 
"""

# Author: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
# Copyright (c) 2009-2011, Emmanuelle Gouillart
# License: BSD

import warnings

import numpy as np
from scipy import sparse, ndimage
try: 
    from scipy.sparse.linalg.dsolve import umfpack
    u = umfpack.UmfpackContext()
except:
    warnings.warn("""Scipy was built without UMFPACK. Consider rebuilding 
    Scipy with UMFPACK, this will greatly speed up the random walker 
    functions. You may also install pyamg and run the random walker function 
    in cg_mg mode (see the docstrings)
    """)
try:
    from pyamg import smoothed_aggregation_solver, ruge_stuben_solver
    amg_loaded = True
except ImportError:
    amg_loaded = False 
import scipy
from scipy.sparse.linalg import cg
try:
    import numexpr as ne
    numexpr_loaded = True
except ImportError:
    numexpr_loaded = False

#-----------Laplacian--------------------

def _make_edges_3d(n_x, n_y, n_z):
    """ 
    Returns a list of edges for a 3D image.
    
    Parameters
    ===========
    n_x: integer
        The size of the grid in the x direction.
    n_y: integer
        The size of the grid in the y direction
    n_z: integer
        The size of the grid in the z direction
    """
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(),
                            vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges

def _compute_weights_3d(edges, data, beta=130, eps=1.e-6):
    l_x, l_y, l_z = data.shape
    gradients = _compute_gradients_3d(data)**2
    beta /= 10 * data.std()
    gradients *= beta
    if numexpr_loaded:
        weights = ne.evaluate("exp(- gradients)") 
    else:
        weights = np.exp(- gradients)
    weights += eps
    return weights

def _compute_gradients_3d(data):
    l_x, l_y, l_z = data.shape
    gr_deep = np.abs(data[:, :, :-1] - data[:, :, 1:]).ravel()
    gr_right = np.abs(data[:, :-1] - data[:, 1:]).ravel()
    gr_down = np.abs(data[:-1] - data[1:]).ravel()
    return np.r_[gr_deep, gr_right, gr_down]

def _make_laplacian_sparse(edges, weights):
    """
    Sparse implementation
    """
    pixel_nb = edges.max() + 1
    diag = np.arange(pixel_nb)
    i_indices = np.hstack((edges[0], edges[1]))
    j_indices = np.hstack((edges[1], edges[0]))
    data = np.hstack((-weights, -weights))
    lap = sparse.coo_matrix((data, (i_indices, j_indices)), 
                            shape=(pixel_nb, pixel_nb))
    connect = - np.ravel(lap.sum(axis=1))
    lap = sparse.coo_matrix((np.hstack((data, connect)),
                (np.hstack((i_indices,diag)), np.hstack((j_indices, diag)))), 
                shape=(pixel_nb, pixel_nb))
    return lap.tocsr()

def _clean_labels_ar(X, labels):
    labels = np.ravel(labels)
    labels[labels == 0] = X
    return labels

def _buildAB(lap_sparse, labels):
    """
        Build the matrix A and rhs B of the linear system to solve
    """
    l_x, l_y, l_z = labels.shape
    labels = labels[labels >= 0]
    indices = np.arange(labels.size) 
    unlabeled_indices = indices[labels == 0] # 有label的节点的索引值
    seeds_indices = indices[labels > 0] # 有label的节点的索引值
    # The following two lines take most of the time
    B = lap_sparse[unlabeled_indices][:, seeds_indices]
    lap_sparse = lap_sparse[unlabeled_indices][:, unlabeled_indices]
    nlabels = labels.max()
    rhs = []
    for lab in range(1, nlabels+1):
        mask = (labels[seeds_indices] == lab)
        fs = sparse.csr_matrix(mask)
        fs = fs.transpose()
        rhs.append(B * fs)
    return lap_sparse, rhs

def _trim_edges_weights(edges, weights, mask):
    mask0 = np.hstack((mask[:, :, :-1].ravel(), mask[:, :-1].ravel(), 
                        mask[:-1].ravel()))
    mask1 = np.hstack((mask[:, :, 1:].ravel(), mask[:, 1:].ravel(),
                            mask[1:].ravel()))
    
    ind_mask = np.logical_and(mask0, mask1)
    edges, weights = edges[:, ind_mask], weights[ind_mask]
    maxval = edges.max()
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(maxval+1))
    edges = order[edges]
    return edges, weights

def _build_laplacian(data, mask=None, beta=50):
    l_x, l_y, l_z = data.shape
    edges = _make_edges_3d(l_x, l_y, l_z)
    weights = _compute_weights_3d(edges, data, beta=beta, eps=1.e-10)
    if mask is not None:
        edges, weights = _trim_edges_weights(edges, weights, mask)
    lap =  _make_laplacian_sparse(edges, weights)
    del edges, weights
    return lap



#----------- Random walker algorithms (with markers or with prior) -------------

def random_walker(data, labels, beta=130, mode='cg_mg', tol=1.e-3, copy=True):
    """
        Segmentation with random walker algorithm by Leo Grady, 
        given some data and an array of labels (the more labeled 
        pixels, the less unknowns and the faster the resolution)

        Parameters
        ----------

        data : array_like
            Image to be segmented in regions. `data` can be two- or
            three-dimensional.

        labels : array of ints
            Array of seed markers labeled with different integers
            for different phases. Negative labels correspond to inactive
            pixels that do not diffuse (they are removed from the graph).

        beta : float
            Penalization coefficient for the random walker motion
            (the greater `beta`, the more difficult the diffusion).

        mode : {'bf', 'cg_mg', 'cg'}
            Mode for solving the linear system in the random walker 
            algorithm. 

            - 'bf' (brute force): an LU factorization of the Laplacian
            is computed. This is fast for small images (<256x256), but
            very slow (due to the memory cost) and memory-consuming for
            big images.

            - 'cg' (conjugate gradient): the linear system is solved
            iteratively using the Conjugate Gradient method from
            scipy.sparse.linalg. This is less memory-consuming than the
            brute force method for large images, but it is quite slow.

            - 'cg_mg' (conjugate gradient with multigrid preconditioner):
            a preconditioner is computed using a multigrid solver, then 
            the solution is computed with the Conjugate Gradient method. 
            This mode requires that the pyamg module 
            (http://code.google.com/p/pyamg/) is installed.

        tol : tolerance to achieve when solving the linear system

        copy : bool
            If copy is False, the `labels` array will be overwritten with
            the result of the segmentation. Use copy=False if you want to 
            save on memory.

        Returns
        -------

        output : ndarray of ints
            Array in which each pixel has been attributed the label number
            that reached the pixel first by diffusion.

        Notes
        -----

        The algorithm was first proposed in *Random walks for image 
        segmentation*, Leo Grady, IEEE Trans Pattern Anal Mach Intell. 
        2006 Nov;28(11):1768-83.

        Examples
        --------

        >>> a = np.zeros((10, 10)) + 0.2*np.random.random((10, 10))
        >>> a[5:8, 5:8] += 1
        >>> b = np.zeros_like(a)
        >>> b[3,3] = 1 #Marker for first phase
        >>> b[6,6] = 2 #Marker for second phase
        >>> random_walker(a, b)
        array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])

    """
    # We work with 3-D arrays
    data = np.atleast_3d(data)
    if copy:
        labels = np.copy(labels)
    labels = labels.astype(np.int)
    # If the array has pruned zones, be sure that no isolated pixels
    # exist between pruned zones (they could not be determined)
    if np.any(labels<0):
        filled = ndimage.binary_propagation(labels>0, mask=labels>=0)
        labels[np.logical_and(np.logical_not(filled), labels == 0)] = -1
        del filled
    labels = np.atleast_3d(labels)
    if np.any(labels < 0):
        lap_sparse = _build_laplacian(data, mask=labels >= 0, beta=beta)
    else:
        lap_sparse = _build_laplacian(data, beta=beta)
    lap_sparse, B = _buildAB(lap_sparse, labels)
    # We solve the linear system
    # lap_sparse X = B
    # where X[i, j] is the probability that a marker of label i arrives 
    # first at pixel j by diffusion
    if mode == 'cg':
        X = _solve_cg(lap_sparse, B, tol=tol)
    if mode == 'cg_mg':
        X = _solve_cg_mg(lap_sparse, B, tol=tol)
    if mode == 'bf':
        X = _solve_bf(lap_sparse, B)
    X = _clean_labels_ar(X + 1, labels)
    data = np.squeeze(data)
    return X.reshape(data.shape)

def _solve_bf(lap_sparse, B): 
    lap_sparse = lap_sparse.tocsc()
    solver = sparse.linalg.factorized(lap_sparse.astype(np.double))
    X = np.array([solver(np.array((-B[i]).todense()).ravel())\
            for i in range(len(B))])
    X = np.argmax(X, axis=0)
    return X

def _solve_cg(lap_sparse, B, tol):
    lap_sparse = lap_sparse.tocsc()
    X = []
    for i in range(len(B)):
        x0 = cg(lap_sparse, -B[i].todense(), tol=tol)[0]
        X.append(x0)
    X = np.array(X)
    X = np.argmax(X, axis=0)
    return X

def _solve_cg_mg(lap_sparse, B, tol):
    if not amg_loaded:
        print( """the pyamg module (http://code.google.com/p/pyamg/)
        must be installed to use the amg mode""")
        raise ImportError
    X = []
    #lap_sparse = lap_sparse.tocsr()
    ml = ruge_stuben_solver(lap_sparse)
    M = ml.aspreconditioner(cycle='V')
    for i in range(len(B)):
        x0 = cg(lap_sparse, -B[i].todense(), tol=tol, M=M, maxiter=30)[0]
        X.append(x0)
    X = np.array(X)
    X = np.argmax(X, axis=0)
    return X


def random_walker_prior(data, prior, mode='bf', gamma=1.e-2):
    """
        Parameters
        ----------

        data : array_like
            Image to be segmented in regions. `data` can be two- or
            three-dimensional.


        prior : array_like
            Array of 1-dimensional probabilities that the pixels 
            belong to the different phases. The size of `prior` is n x s
            where n is the number of phases to be segmented, and s the 
            total number of pixels.

        mode : {'bf', 'amg'}
            Mode for solving the linear system in the random walker 
            algorithm. `mode` can be either 'bf' (for brute force),
            in which case matrices are directly inverted, or 'amg'
            (for algebraic multigrid solver), in which case a multigrid
            approach is used. The 'amg' mode uses the pyamg module 
            (http://code.google.com/p/pyamg/), which must be installed
            to use this mode.

        gamma : float
            gamma determines the absence of confidence into the prior. 
            The smaller gamma, the more the output values will be determined
            according to the prior only. Conversely, the greater gamma, 
            the more continuous the segmented regions will be.

        Returns
        -------

        output : ndarray of ints
            Segmentation of data. The number of phases corresponds to the
            number of lines in prior.

        Notes
        -----

        The algorithm was first proposed in *Multilabel random walker 
        image segmentation using prior models*, L. Grady, IEEE CVPR 2005,
        p. 770 (2005).

        Examples
        --------
        >>> a = np.zeros((40, 40))
        >>> a[10:-10, 10:-10] = 1
        >>> a += 0.7*np.random.random((40, 40))
        >>> p = a.max() - a.ravel()
        >>> q = a.ravel()
        >>> prior = np.array([p, q])
        >>> labs = random_walker_prior(a, prior)
    """
    data = np.atleast_3d(data)
    lap_sparse = _build_laplacian(data, beta=50)
    dia = range(data.size)
    shx, shy = lap_sparse.shape
    lap_sparse = lap_sparse + sparse.coo_matrix(
                        (gamma*prior.sum(axis=0), (range(shx), range(shy))))
    del dia
    if mode == 'bf':
        lap_sparse = lap_sparse.tocsc()
        solver = sparse.linalg.factorized(lap_sparse.astype(np.double))
        X = np.array([solver(gamma*label_prior)
                      for label_prior in prior])
    elif mode == 'amg':
        if not amg_loaded:
            print ("""the pyamg module (http://code.google.com/p/pyamg/)
            must be installed to use the amg mode""")
            raise ImportError
        lap_sparse = lap_sparse.tocsr()
        mls = smoothed_aggregation_solver(lap_sparse)
        del lap_sparse
        X = np.array([mls.solve(gamma*label_prior)
                      for label_prior in prior])
        del mls
    return np.squeeze((np.argmax(X, axis=0)).reshape(data.shape))





 
