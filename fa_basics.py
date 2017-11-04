
import numpy as np, scipy, pandas as pd
import matplotlib.pyplot as plt
import pystan, pystan.external.pymc.plots

def build_toy_dataset(N=300, rs=None):
    if rs == None:
        rs = np.random.RandomState(42)
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
    # Z   = np.random.multivariate_normal(mu_z, sigma_z, size=N).T
    Z   = rs.multivariate_normal(mu_z, sigma_z, size=N).T

    # sample error vector # size(D,N) = size(10,N)
    # epsilon = np.random.multivariate_normal(mu_epsilon, psi, size=N).T
    epsilon = rs.multivariate_normal(mu_epsilon, psi, size=N).T

    X = np.dot(L, Z) + epsilon  # generate observable data # size(D,N) = size(10,N)
    return X, L, Z, psi

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

def extract(mean_values_series, startswith=None, shape=None):
    values = [(mean_values_series.index[i], mean_values_series[i]) for i in range(len(mean_values_series)) if mean_values_series.index[i].startswith(startswith)]
    values_df = pd.DataFrame(values)
    if len(shape) == 1:
        return values_df.values[:,1].astype(np.float64)
    elif len(shape) == 2:
        if shape[0]*shape[1] != len(values_df):
            print('the shape to extract {} does not fit the size of values available {}'.format(shape, len(values_df)))
            return None
        return np.reshape(values_df.values[:,1],shape,order='F').astype(np.float64)

fa_ard_advi_model_string = """
data {
    int<lower=1> N;                // Number of samples 
    int<lower=1> D;                // The original dimension; convert Ps to Ds
    int<lower=1> K;                // The latent dimension; convert Ds to Ks
    matrix[N,D] X;                 // The data matrix data matrix of order [N,D]
}
parameters {
    matrix[N, K] Z; // The latent matrix
    matrix[D, K] L; // The weight matrix
    real<lower=0> tau; // Noise term 
    vector<lower=0>[K] alpha; // ARD prior
}
transformed parameters{
    vector<lower=0>[K] t_alpha;
    real<lower=0> t_tau;
    t_alpha = inv(sqrt(alpha));
    t_tau = inv(sqrt(tau));
}
model {
    tau ~ gamma(1,1);
    to_vector(Z) ~ normal(0,1);
    alpha ~ gamma(1e-3,1e-3);
    for(k in 1:K) L[,k] ~ normal(0, t_alpha[k]);
    to_vector(X) ~ normal(to_vector(Z*L'), t_tau);
}
"""

fa_ard_advi_model_sm = None

# https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
from io import StringIO
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def fa_ard_advi(X):
    global fa_ard_advi_model_sm
    if fa_ard_advi_model_sm == None:
        with Capturing() as output:
            fa_ard_advi_model_sm = pystan.StanModel(model_code=fa_ard_advi_model_string)
    N = X.shape[0]
    D = X.shape[1]
    # K = X.shape[1]

    data_list = dict(N = N, D = D, K = D, X = X)

    fit = fa_ard_advi_model_sm.vb(data=data_list, algorithm='meanfield', output_samples=10000)
    advi_samples_df = pd.read_csv(fit['args']['sample_file'].decode('ascii'), comment='#')
    mean_values_series = advi_samples_df.mean(axis=0)
    return mean_values_series

# http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html
# adapted from http://matplotlib.org/examples/specialty_plots/hinton_demo.html
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('lightgray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'red' if w > 0 else 'blue'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    nticks = matrix.shape[0]
    ax.xaxis.tick_top()
    ax.set_xticks(range(nticks))
    # ax.set_xticklabels(list(matrix.columns), rotation=90)
    ax.set_xticklabels(list(range(matrix.shape[1])), rotation=90)
    ax.set_yticks(range(nticks))
    #ax.set_yticklabels(matrix.columns)
    ax.set_yticklabels(list(range(matrix.shape[1])))
    ax.grid(False)

    ax.autoscale_view()
    ax.invert_yaxis()

# http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html
def calcWithinGroupsVariance(variable, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    # get the mean and standard deviation for each group:
    numtotal = 0
    denomtotal = 0
    for leveli in levels:
        levelidata = variable[groupvariable==leveli]
        levelilength = len(levelidata)
        # get the standard deviation for group i:
        sdi = np.std(levelidata)
        numi = (levelilength)*sdi**2
        denomi = levelilength
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    # calculate the within-groups variance
    Vw = numtotal / (denomtotal - numlevels)
    return Vw

def calcWithinGroupsCovariance(variable1, variable2, groupvariable):
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    Covw = 0.0
    # get the covariance of variable 1 and variable 2 for each group:
    for leveli in levels:
        levelidata1 = variable1[groupvariable==leveli]
        levelidata2 = variable2[groupvariable==leveli]
        mean1 = np.mean(levelidata1)
        mean2 = np.mean(levelidata2)
        levelilength = len(levelidata1)
        # get the covariance for this group:
        term1 = 0.0
        for levelidata1j, levelidata2j in zip(levelidata1, levelidata2):
            term1 += (levelidata1j - mean1)*(levelidata2j - mean2)
        Cov_groupi = term1 # covariance for this group
        Covw += Cov_groupi
    totallength = len(variable1)
    Covw /= totallength - numlevels
    return Covw

# http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html
def calcBetweenGroupsVariance(variable, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set((groupvariable)))
    numlevels = len(levels)
    # calculate the overall grand mean:
    grandmean = np.mean(variable)
    # get the mean and standard deviation for each group:
    numtotal = 0
    denomtotal = 0
    for leveli in levels:
        levelidata = variable[groupvariable==leveli]
        levelilength = len(levelidata)
        # get the mean and standard deviation for group i:
        meani = np.mean(levelidata)
        sdi = np.std(levelidata)
        numi = levelilength * ((meani - grandmean)**2)
        denomi = levelilength
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    # calculate the between-groups variance
    Vb = numtotal / (numlevels - 1)
    return(Vb)

def calcBetweenGroupsCovariance(variable1, variable2, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    # calculate the grand means
    variable1mean = np.mean(variable1)
    variable2mean = np.mean(variable2)
    # calculate the between-groups covariance
    Covb = 0.0
    for leveli in levels:
        levelidata1 = variable1[groupvariable==leveli]
        levelidata2 = variable2[groupvariable==leveli]
        mean1 = np.mean(levelidata1)
        mean2 = np.mean(levelidata2)
        levelilength = len(levelidata1)
        term1 = (mean1 - variable1mean) * (mean2 - variable2mean) * levelilength
        Covb += term1
    Covb /= numlevels - 1
    return Covb
