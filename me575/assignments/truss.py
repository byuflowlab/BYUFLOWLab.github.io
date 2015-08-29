import numpy as np


def bar(E, A, L, phi):
    """Compute the stiffness and stress matrix for one element

    Parameters
    ----------
    E : float
        modulus of elasticity
    A : float
        cross-sectional area
    L : float
        length of element
    phi : float
        orientation of element

    Outputs
    -------
    K : 4 x 4 ndarray
        stiffness matrix
    S : 1 x 4 ndarray
        stress matrix
    dKdA : 4 x 4 ndarray
        derivative of stiffness matrix w.r.t. A

    """

    # Your code goes here

    return K, S, dKdA



def node2idx(node, DOF):
    """You do not need to change this function
    It computes the appropriate indices in the global matrix for
    the corresponding node numbers.  You pass in the number of the node
    (either as a scalar or an array of locations), and the degrees of
    freedom per node and it returns the corresponding indices in
    the global matrices

    """

    idx = np.array([], dtype=np.int)

    for i in range(len(node)):

        n = node[i]
        start = DOF*(n-1)
        finish = DOF*n

        idx = np.concatenate((idx, np.arange(start, finish, dtype=np.int)))

    return idx


def truss(A):
    """This is the subroutine for the 10-bar truss

    Parameters
    ----------
    A : ndarray of length 10
        cross-sectional areas of all the bars

    Outputs
    -------
    mass : float
        mass of the entire structure
    stress : ndarray of length 10
        stress of each bar

    Optional (You may want to compute analytic derivatives in this routine,
        or you may want to do it in another routine, finite differencing
        or complex step cannot return the below gradients from this routine
        it requires a separate routine)
    -------
    dmass_dA : ndarray of length 10
        derivative of mass w.r.t. each A
    dstress_dA : 10 x 10 ndarray
        dstress_dA[i, j] is derivative of stress[i] w.r.t. A[j]

    """


    # initialization
    n = 6  # number of nodes
    DOF = 2  # number of degrees of freedom per node
    nbar = 10  # number of bars

     # define constants
     # TODO

    # TODO: define 10-bar truss (each should be vectors of length 10)
    start = []  # idx of starting node
    finish = []  # idx of finish node
    L = []  # length of element
    phi = []  # angle of element

    # compute mass
    # TODO


    # assemble global matricies
    K = np.zeros((DOF*n, DOF*n))
    S = np.zeros((nbar, DOF*n))

    for i in range(nbar):  # loop through each bar

        # compute the submatrices for the element
        Ksub, Ssub, _ = bar(E, A[i], L[i], phi[i])

        # insert submatrix into global matrix
        idx = node2idx([start[i], finish[i]], DOF)  # pass in the starting and ending node number for this element
        K[np.ix_(idx, idx)] += Ksub
        S[i, idx] = Ssub


    # setup applied loads
    F = np.zeros((n*DOF, 1))

    idx = node2idx([2], DOF)
    F[idx[1]] = -P
    idx = node2idx([4], DOF)
    F[idx[1]] = -P


    # setup boundary condition
    rigid = [5, 6]
    remove = node2idx(rigid, DOF)

    K = np.delete(K, remove, axis=0)
    K = np.delete(K, remove, axis=1)
    F = np.delete(F, remove, axis=0)
    S = np.delete(S, remove, axis=1)


    # solve for deflections
    # TODO

    # compute stress
    # TODO

    # compute analytic gradients (direct or adjoint)
    # TODO
    # note that finite difference and complex step cannot be done here,
    # those can only be computed external to this function.


    return mass, stress, dmass_dA, dstress_dA
