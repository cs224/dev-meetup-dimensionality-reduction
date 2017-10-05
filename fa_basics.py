
import numpy as np, scipy

def build_toy_dataset(N=300):
    D=10
    K=3
    mu_z    = np.zeros(K) # the mean of theta
    sigma_z = np.diag(np.ones(K))

    mu_epsilon = np.zeros(D) # the mean of epsilon
    psi = np.diag([0.2079, 0.19, 0.1525, 0.20, 0.36, 0.1875, 0.1875, 1.00, 0.27, 0.27])

    l1  = np.array([0.99, 0.00, 0.25, 0.00, 0.80, 0.00, 0.50, 0.00, 0.00, 0.00]).reshape(-1,1)
    l2  = np.array([0.00, 0.90, 0.25, 0.40, 0.00, 0.50, 0.00, 0.00, -0.30, -0.30]).reshape(-1,1)
    l3  = np.array([0.00, 0.00, 0.85, 0.80, 0.00, 0.75, 0.75, 0.00, 0.80, 0.80]).reshape(-1,1)
    L = np.hstack([l1, l2, l3]) # size(10,3)

    # sample factor scores # size(K,N) = size(3,N)
    Z   = np.random.multivariate_normal(mu_z, sigma_z, size=N).T
    # sample error vector # size(D,N) = size(10,N)
    epsilon = np.random.multivariate_normal(mu_epsilon, psi, size=N).T

    X = np.dot(L, Z) + epsilon  # generate observable data # size(D,N) = size(10,N)
    return X, L, Z

# build_toy_dataset(N=12) # size(10,12)

def calc_principal_angles(matrix1, matrix2):
    """
    Calculates the principal angles between `matrix1` and `matrix2`.

    Parameters
    ----------
    matrix1 : np.ndarray
        A 2D numpy array.
    matrix2 : np.ndarray
        A 2D numpy array.

    Returns
    -------
    np.ndarray
        The principal angles between `matrix1` and `matrix2`. This is a
        1D numpy array.

    See also
    --------
    calc_chordal_distance_from_principal_angles

    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4], [5, 6]])
    >>> B = np.array([[1, 5], [3, 7], [5, -1]])
    >>> print(calc_principal_angles(A, B))
    [ 0.          0.54312217]
    """
    # First we need to find the orthogonal basis for matrix1 and
    # matrix2. This can be done with the QR decomposition. Note that if
    # matrix1 has 'n' columns then its orthogonal basis is given by the
    # first 'n' columns of the 'Q' matrix from its QR decomposition.
    Q1 = np.linalg.qr(matrix1)[0]
    Q2 = np.linalg.qr(matrix2)[0]

    # TODO: Test who has more columns. Q1 must have dimension grater than
    # or equal to Q2 so that the SVD can be calculated in the order below.
    #
    # See the algorithm in
    # http://sensblogs.wordpress.com/2011/09/07/matlab-codes-for-principal-angles-also-termed-as-canonical-correlation-between-any-arbitrary-subspaces-redirected-from-jen-mei-changs-dissertation/
    S = np.linalg.svd(
        Q1.conjugate().transpose().dot(Q2), full_matrices=False)[1]

    # The singular values of S vary between 0 and 1, but due to
    # computational impressions there can be some value above 1 (by a very
    # small value). Below we change values greater then 1 to be equal to 1
    # to avoid problems with the arc-cos call later.
    S[S > 1] = 1  # Change values greater then 1 to 1

    # The singular values in the matrix S are equal to the cosine of the
    # principal angles. We can calculate the arc-cosine of each element
    # then.
    return np.arccos(S)

def projection_matrix(A):
    inv_AT_A = scipy.linalg.inv(np.dot(A.T, A))
    P = np.dot(np.dot(A, inv_AT_A),A.T)
    return P

def grassmannian_norm(M1, M2, type=2):
    return scipy.linalg.norm(projection_matrix(M1) - projection_matrix(M2), type)