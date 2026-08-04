import numpy as np
import scipy as sp
from scipy.special import logsumexp
import cvxpy as cp
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import os

############################################################################################################
################## Model Fitting ###########################################################################
############################################################################################################

def iso(Y,u = None,beta= None,alpha = None,lam = None,verbose = False):
    '''
    The interaction screening objective for a generalized Potts model.

    Parameters
    ----------
    Y : np.array
        The data matrix of size n x p of angles. p is the number of nodes and n is the number of samples.
    u : int
        The node whose interactions are to be estimated
    beta : float
        The maximum coupling parameter. If None, beta is infinity.
    alpha : float
        The minimum coupling parameter. If None, alpha is set to 0.
    lam : float
        The penalty parameter. If None, lam is set to 1.

    Returns
    -------
    thetas_u_opt : np.array
        The optimal coupling parameters for the u-th node.
    '''

    n,p = Y.shape
    if lam is None:
        epsilon = 0.1
        lam = 4*np.sqrt(np.log(8*p**2/epsilon)/n)

    # Define the optimization variables
    thetas_u = cp.Variable(2*(p-1))

    # Define the constraints
    constraints = []
    if beta is not None:
        constraints.append(cp.abs(thetas_u) <= beta)
        # constraints.append(cp.sqrt(cp.power(thetas_u[:p-1],2) + cp.power(thetas_u[p-1:],2)) <= beta)
    
    # Define the objective
    # 1/n * sum_1^n exp(-sum_alledgesconnectedtou theta_i*cos)

    #genereate a u offset data matrix for which each column is the difference between the u-th node and the i-th node if u > i and i-th node and the u-th node if i > u
    Y_offset = np.zeros((n,p))
    for i in range(p):
        if i < u:
            Y_offset[:,i] = Y[:,u] - Y[:,i]
        elif i > u:
            Y_offset[:,i] = Y[:,i] - Y[:,u]
    
    Y_offset = np.delete(Y_offset,u,axis=1)
    cos_Y = np.cos(Y_offset)
    sin_Y = np.sin(Y_offset)

    Z = np.hstack([cos_Y,sin_Y])

    # Generate the objective

    Sn = 1/n * cp.sum(cp.exp(-Z @ thetas_u)) 
    th_c = cp.reshape(thetas_u[:p-1],(p-1,1))
    th_s = cp.reshape(thetas_u[p-1:],(p-1,1))
    th = cp.hstack([th_c,th_s])

    #obj = Sn + lam * cp.norm(thetas_u,1)
    obj = Sn + lam * cp.norm(cp.norm(th,2,axis = 1),1)
    prob = cp.Problem(cp.Minimize(obj))#,constraints)

    try:
        prob.solve(solver=cp.MOSEK,verbose=verbose)
    except:
        print('MOSEK failed')
        prob.solve(solver=cp.SCS,verbose=verbose)
    
    thetas_u_opt = thetas_u.value

    #set the entries with small values to 0
    if alpha is not None:
        idces = np.sqrt(thetas_u_opt[:p-1]**2 + thetas_u_opt[p-1:]**2) < alpha/2
        idces = np.concatenate([idces,idces])
        thetas_u_opt[idces] = 0
        # thetas_u_opt[np.abs(thetas_u_opt) < alpha/2] = 0 #use this if you want to treat cosine and sine terms separately

    return thetas_u_opt





def fit(Y,beta= None,alpha = None,lam = None,parallel = False,verbose = False):
    '''
    Fit the generalized Potts model to the data. Solve for parameters using interaction screening.

    Parameters
    ----------
    Y : np.array
        The data matrix of size n x p of angles. p is the number of nodes and n is the number of samples.
    beta : float
        The maximum coupling parameter. If None, beta is infinity.
    alpha : float
        The minimum coupling parameter. If None, alpha is set to 0.
    lam : float
        The penalty parameter. If None, lam is set to default.
    
    Returns
    -------
    thetas : list
        The optimal coupling parameters for the p nodes.
        thetas[0] are the coupling parameters for the cosine terms and thetas[1] are the coupling parameters for the sine terms.
    '''

    n,p = Y.shape
    theta_c_hat = np.zeros((p,p))
    theta_s_hat = np.zeros((p,p))
    thetas_dict = {}

    print('Fitting model')
    if parallel:
        try:
            ncpus = int(os.environ['SLURM_CPUS_ON_NODE'])
        except:
            ncpus = cpu_count()
            ncpus = np.min([ncpus,p])
        print(f'Using {ncpus} cpus')

        try:
            results = Parallel(n_jobs=ncpus)(delayed(iso)(Y,u,beta,alpha,lam,verbose) for u in range(p))
            for u in range(p):
                thetas_u_opt = results[u]
                # thetas_u_opt = iso(Y,u,d,beta,alpha,lam)
                thetas_dict[u] = thetas_u_opt
        except:
            print('Parallel failed')
            for u in range(p):
                thetas_u_opt = iso(Y,u,beta,alpha,lam,verbose)
                thetas_dict[u] = thetas_u_opt
    else:
        for u in range(p):
            thetas_u_opt = iso(Y, u, beta, alpha, lam,verbose)
            thetas_dict[u] = thetas_u_opt
    
    for u in range(p):
        #fill in upper triangular part of theta_c_hat and theta_s_hat
        #theta_c_hat is symmetric and theta_s_hat is skew-symmetric

        for i in range(p):
            if i < u:
                theta_c_hat[i,u] += thetas_dict[u][i]
                theta_s_hat[i,u] += thetas_dict[u][i+p-1]
            elif i > u:
                theta_c_hat[u,i] += thetas_dict[u][i-1]
                theta_s_hat[u,i] += thetas_dict[u][i-1+p-1]
    
    theta_c_hat = theta_c_hat/2
    theta_s_hat = theta_s_hat/2
    theta_c_hat = theta_c_hat + theta_c_hat.T
    theta_s_hat = theta_s_hat - theta_s_hat.T

    thetas = [theta_c_hat,theta_s_hat]

    return thetas

############################################################################################################
################## Sparse Model Fitting ####################################################################
############################################################################################################





## The following code is for the case where the active set is the same for all nodes

def iso_sparse(Y,active_set,u = None,beta= None,alpha = None,verbose = False):
    '''
    The interaction screening objective for a generalized Potts model.
    Only optimize over the active set of nodes.

    Parameters
    ----------
    Y : np.array
        The data matrix of size n x p of angles. p is the number of nodes and n is the number of samples.
    active_set : list
        The list of interacting nodes to optimize over.
    u : int
        The node whose interactions are to be estimated
    beta : float
        The maximum coupling parameter. If None, beta is infinity.
    alpha : float
        The minimum coupling parameter. If None, alpha is set to 0.
    
    Returns
    -------
    thetas_u_opt : np.array
        The optimal coupling parameters for the u-th node.
    '''

    n,p = Y.shape

    # Define the optimization variables
    d = len(active_set)
    if d == 0:
        return np.zeros(0)
    thetas_u = cp.Variable(2*d)

    # Define the constraints
    constraints = []
    if beta is not None:
        constraints.append(cp.abs(thetas_u) <= beta)
    
    # Define the objective

    #genereate a u offset data matrix for which each column is the difference between the u-th node and the i-th node if u > i and i-th node and the u-th node if i > u
    Y_offset = np.zeros((n,d))
    Y_u = Y[:,u]
    Y_active = Y[:,active_set]
    for i in range(d):
        if active_set[i] < u:
            Y_offset[:,i] = Y_u - Y_active[:,i]
        elif active_set[i] > u:
            Y_offset[:,i] = Y_active[:,i] - Y_u
        # if active_set[i] < u:
        #     Y_offset[:,i] = Y[:,u] - Y[:,active_set[i]]
        # elif active_set[i] > u:
        #     Y_offset[:,i] = Y[:,active_set[i]] - Y[:,u]

    cos_Y = np.cos(Y_offset)
    sin_Y = np.sin(Y_offset)

    Z = np.hstack([cos_Y,sin_Y])

    # Generate the objective
    Sn = 1/n * cp.sum(cp.exp(-Z @ thetas_u))
    prob = cp.Problem(cp.Minimize(Sn))

    try:
        prob.solve(solver=cp.MOSEK,verbose=verbose)
    except:
        print('MOSEK failed')
        prob.solve(solver=cp.SCS,verbose=verbose)

    thetas_u_opt = thetas_u.value
    return thetas_u_opt


def fit_sparse(Y,beta= None,alpha = None,lam = None,parallel = False,verbose = False):
    '''
    Fit the generalized Potts model to the data. Solve for parameters using interaction screening.
    First solves the problem with a large penalty parameter and then solves the problem with no penalty.

    Parameters
    ----------
    Y : np.array
        The data matrix of size n x p of angles. p is the number of nodes and n is the number of samples.
    beta : float
        The maximum coupling parameter. If None, beta is infinity.
    alpha : float
        The minimum coupling parameter. If None, alpha is set to 0.
    lam : float
        The penalty parameter. If None, lam is set to default.

    Returns
    -------
    thetas : list
        The optimal coupling parameters for the p nodes.
        thetas[0] are the coupling parameters for the cosine terms and thetas[1] are the coupling parameters for the sine terms.
    '''

    thetas_c_reg,thetas_s_reg = fit(Y,beta,alpha,lam,parallel,verbose)
    kappas = np.sqrt(thetas_c_reg**2 + thetas_s_reg**2)

    n,p = Y.shape
    theta_c_hat = np.zeros((p,p))
    theta_s_hat = np.zeros((p,p))
    thetas_dict = {}
    active_sets = []

    #find the nodes with non-zero coupling parameters
    for i in range(p):
        active_set = np.where(kappas[i] > 0)[0]
        active_sets.append(active_set)
    print('Fitting sparse model')

    if parallel:
        try:
            ncpus = int(os.environ['SLURM_CPUS_ON_NODE'])
        except:
            ncpus = cpu_count()
            ncpus = np.min([ncpus,p])
        print(f'Using {ncpus} cpus')

        results = Parallel(n_jobs=ncpus)(delayed(iso_sparse)(Y,active_sets[u],u,beta,alpha,verbose) for u in range(p))
        for u in range(p):
            thetas_dict[u] = results[u]
    else:
        for u in range(p):
            thetas_dict[u] = iso_sparse(Y,active_sets[u],u,beta,alpha,verbose)

    for u in range(p):
        #fill in upper triangular part of theta_c_hat and theta_s_hat
        #theta_c_hat is symmetric and theta_s_hat is skew-symmetric
        d = len(active_sets[u])
        for idx,i in enumerate(active_sets[u]):
            if i < u:
                theta_c_hat[i,u] += thetas_dict[u][idx]
                theta_s_hat[i,u] += thetas_dict[u][idx+d]
            elif i > u:
                theta_c_hat[u,i] += thetas_dict[u][idx]
                theta_s_hat[u,i] += thetas_dict[u][idx+d]


            # if i < u:
            #     theta_c_hat[i,u] += thetas_dict[u][i]
            #     theta_s_hat[i,u] += thetas_dict[u][i+p-1]
            # elif i > u:
            #     theta_c_hat[u,i] += thetas_dict[u][i-1]
            #     theta_s_hat[u,i] += thetas_dict[u][i-1+p-1]

    theta_c_hat = theta_c_hat/2
    theta_s_hat = theta_s_hat/2
    theta_c_hat = theta_c_hat + theta_c_hat.T
    theta_s_hat = theta_s_hat - theta_s_hat.T

    thetas = [theta_c_hat,theta_s_hat]

    return thetas

def dynamic_iso():
    pass


############################################################################################################
################## Gibbs Sampling ##########################################################################
############################################################################################################

def gibbs(thetas,n_samples = 1000,burn_in = 100,Q = 1,fsave = None):
    '''
    Perform Gibbs sampling to sample from the posterior distribution of the generalized Potts model.

    Parameters
    ----------
    thetas : list
        The coupling parameters for the p nodes.
        thetas[0] are the coupling parameters for the cosine terms and thetas[1] are the coupling parameters for the sine terms.
    n_samples : int
        The number of samples to generate.
    burn_in : int
        The number of burn-in samples.
    Q : int
        The number of samples to generate between each sample.
    fsave : str
        The file to save the samples to. If None, do not save the samples.

    Returns
    -------
    samples : np.array
        The samples from the distribution of the generalized Potts model.
    '''

    ## write code to sample from univariate conditionals
    # univariate conditional for each node is a von Mises distribution with some mean and concentration parameter
    # look at overleaf document for details
    p = thetas[0].shape[0]
    
    theta_c = thetas[0]
    theta_s = thetas[1]

    mus = np.arctan2(theta_s,theta_c)
    kappas = np.sqrt(theta_c**2 + theta_s**2)

    samples = []#np.zeros((n_samples*Q,p))
    max_samples = int(2*10e5*np.log(2*p))
    save_every = 10000

    for _ in range(n_samples*Q + burn_in):
        if _ == 0:
            prev_sample = np.random.uniform(-np.pi,np.pi,p) #generate_initial_gibbs_sample(mus,kappas)  
            #compute exp term for last node
            exp_terms = np.array([kappas[p-1,k]*np.exp(1j*(prev_sample[k] - mus[p-1,k])) for k in range(p-1) if k != p-1] )
            Au = np.abs(np.sum(exp_terms))
            xi_u = np.angle(np.sum(exp_terms))
            sample_u = np.random.vonmises(xi_u,Au)
            prev_sample[p-1] = sample_u


        for u in range(p):
            Y_no_u = np.delete(prev_sample,u)
            exp_terms = np.array([kappas[u,k]*np.exp(1j*(prev_sample[k] - mus[u,k])) for k in range(p) if k != u] )
            Au = np.abs(np.sum(exp_terms))
            xi_u = np.angle(np.sum(exp_terms))
            sample_u = np.random.vonmises(xi_u,Au)
            prev_sample[u] = sample_u

        if _ >= burn_in:
            #samples.append(np.copy(prev_sample))
            post_burn = _ - burn_in
            samples.append(np.copy(prev_sample))

            if (post_burn+1) % (save_every*Q) == 0 and fsave is not None:
                print(f'Saving samples: {post_burn//Q}')
                if os.path.exists(fsave):
                    existing_mat = sp.io.loadmat(fsave)
                    existing_samples = existing_mat['samples']
                    if existing_samples.shape[0] < max_samples:
                        to_save = np.array(samples)[-save_every*Q::Q]
                        to_save = np.vstack([existing_samples,to_save])
                        sp.io.savemat(fsave,{'samples':to_save,'theta_c':theta_c,'theta_s':theta_s})
                    else: #break out of whole loop
                        break
                else:
                    sp.io.savemat(fsave,{'samples':np.array(samples)[::Q],'theta_c':theta_c,'theta_s':theta_s})
                    

    samples = np.array(samples)
    samples = samples[::Q]

    return samples

############################################################################################################
################## Likelihood Computations #################################################################
############################################################################################################


def unnormalized_likelihood(Y, thetas):
    '''
    Compute the unnormalized likelihood associated with the graphical model for a set of samples

    Parameters:
    ---
    Y : np.array
        The data matrix of size n x p of angles. p is the number of nodes and n is the number of samples.
    thetas : list
        The coupling parameters for the p nodes.

    Returns:
    ---
    unnormalized_likelihood : float
        The unnormalized likelihood of the data parameterized by the coupling parameters.
    '''
    thetas_c = thetas[0]
    thetas_s = thetas[1]
    p = thetas_c.shape[0]
    mus = np.arctan2(thetas_s, thetas_c)
    kappas = np.sqrt(thetas_c**2 + thetas_s**2)

    exp_terms = [ ]

    for i in range(p):
        for j in range(i+1,p):
            tmp = kappas[i,j]*np.cos(Y[:,j] - Y[:,i] - mus[i,j])
            exp_terms.append(tmp)
    
    exp_terms = np.array(exp_terms)
    unnormalized_likelihoods = np.exp(np.sum(exp_terms,axis=0))
            
    return unnormalized_likelihoods

def unnormalized_log_likelihood(Y, thetas):
    '''
    Compute the unnormalized likelihood associated with the graphical model for a set of samples

    Parameters:
    ---
    Y : np.array
        The data matrix of size n x p of angles. p is the number of nodes and n is the number of samples.
    thetas : list
        The coupling parameters for the p nodes.

    Returns:
    ---
    unnormalized_likelihood : float
        The unnormalized likelihood of the data parameterized by the coupling parameters.
    '''
    thetas_c = thetas[0]
    thetas_s = thetas[1]
    p = thetas_c.shape[0]
    mus = np.arctan2(thetas_s, thetas_c)
    kappas = np.sqrt(thetas_c**2 + thetas_s**2)

    exp_terms = [ ]

    for i in range(p):
        for j in range(i+1,p):
            tmp = kappas[i,j]*np.cos(Y[:,j] - Y[:,i] - mus[i,j])
            exp_terms.append(tmp)

    # i, j = np.triu_indices(p, k=1)
    
    # # Compute the cosine terms in a vectorized manner
    # tmp = kappas[i, j] * np.cos(Y[:, j] - Y[:, i] - mus[i, j])
    
    
    exp_terms = np.array(exp_terms)
    unnormalized_log_likelihoods = np.sum(exp_terms,axis=0)

    # i_indices, j_indices = np.triu_indices(p, 1) # Get upper triangle indices, offset by 1 to skip diagonal.

    # diffs = Y[:, j_indices] - Y[:, i_indices]
    # terms = kappas[i_indices, j_indices] * np.cos(diffs - mus[i_indices, j_indices])

    # unnormalized_log_likelihoods = np.sum(terms, axis=1)
            
    return unnormalized_log_likelihoods

def log_likelihood(Y, thetas, log_Z= None):
    '''
    Compute the likelihood associated with the graphical model for a set of samples

    Parameters:
    ---
    Y : np.array
        The data matrix of size n x p of angles. p is the number of nodes and n is the number of samples.
    thetas : list
        The coupling parameters for the p nodes.

    Returns:
    ---
    likelihood : float
        The likelihood of the data parameterized by the coupling parameters.
    '''
    if log_Z is None:
        log_Z = log_partition(thetas, n_samples = int(5e4*np.log(Y.shape[1])))
    unnormalized_log_likelihoods = unnormalized_log_likelihood(Y, thetas)
    log_likelihoods = unnormalized_log_likelihoods - log_Z
    log_likelihood = np.sum(log_likelihoods)
    return log_likelihood


def log_partition(thetas, n_samples = None):
    p = thetas[0].shape[0]
    if n_samples is None:
        n_samples = int(5e4*np.log(p))
    uniform_samples = np.random.uniform(-np.pi,np.pi,(n_samples,p))
    uniform_density = 1/((2*np.pi)**p)

    tmp = unnormalized_log_likelihood(uniform_samples, thetas) - np.log(uniform_density)
    log_Z = sp.special.logsumexp(tmp) - np.log(n_samples)
    return log_Z

















## these are wrong i think

# def log_likelihood(Y,thetas):
#     '''
#     Compute the log likelihood associated with the graphical model for a set of samples

#     Parameters:
#     ---
#     Y : np.array
#         The data matrix of size n x p of angles. p is the number of nodes and n is the number of samples.
#     thetas : list
#         The coupling parameters for the p nodes.

#     Returns:
#     ---
#     log_likelihood : float
#         The log likelihood of the data parameterized by the coupling parameters.
#     '''
#     thetas_c = thetas[0]
#     thetas_s = thetas[1]
#     p = thetas_c.shape[0]
#     mus = np.arctan2(thetas_s,thetas_c)
#     kappas = np.sqrt(thetas_c**2 + thetas_s**2)

#     ## compute the log partition function using the log sum exp trick
#     #create a p-dimensional mesh to integrate over
#     J = 20
#     Delta = (2*np.pi/J)**p
#     grid_1d = np.linspace(-np.pi,np.pi,J,endpoint = False)+np.pi/J #midpoints of the bins
#     grid = np.meshgrid(*[grid_1d for _ in range(p)])
#     mesh = np.vstack([g.ravel() for g in grid]).T

#     #compute the log partition function
#     log_partition = logsumexp([np.sum([kappas[i,j]*np.cos(mesh[:,j] - mesh[:,i] - mus[i,j])*Delta for i in range(p) for j in range(i+1,p)],axis = 0) for _ in range(Y.shape[0])])

#     #compute the log likelihood
#     log_likelihood = np.sum([np.sum([kappas[i,j]*np.cos(Y[:,j] - Y[:,i] - mus[i,j]) for i in range(p) for j in range(i+1,p)],axis = 0) - log_partition for _ in range(Y.shape[0])])

#     return log_likelihood

# def unnormalized_log_likelihood(Y,thetas):
#     '''
#     Compute the unnormalized log likelihood associated with the graphical model for a set of samples

#     Parameters:
#     ---
#     Y : np.array
#         The data matrix of size n x p of angles. p is the number of nodes and n is the number of samples.
#     thetas : list
#         The coupling parameters for the p nodes.

#     Returns:
#     ---
#     unnormalized_log_likelihood : float
#         The unnormalized log likelihood of the data parameterized by the coupling parameters.
#     '''
#     thetas_c = thetas[0]
#     thetas_s = thetas[1]
#     p = thetas_c.shape[0]
#     mus = np.arctan2(thetas_s, thetas_c)
#     kappas = np.sqrt(thetas_c**2 + thetas_s**2)

#     # unnormalized_log_likelihood = np.sum([
#     #     np.sum([
#     #         kappas[i, j] * np.cos(Y[:, j] - Y[:, i] - mus[i, j])
#     #         for i in range(p) for j in range(i + 1, p)
#     #     ], axis=0)
#     #     for _ in range(Y.shape[0])
#     # ])
#     Y_diff = Y[:, :, np.newaxis] - Y[:, np.newaxis, :]
#     cos_terms = np.cos(Y_diff - mus)
#     unnormalized_log_likelihood = np.sum(kappas * cos_terms)

#     return unnormalized_log_likelihood

# def unnormalized_likelihood(Y, thetas):
#     '''
#     Compute the unnormalized likelihood associated with the graphical model for a set of samples

#     Parameters:
#     ---
#     Y : np.array
#         The data matrix of size n x p of angles. p is the number of nodes and n is the number of samples.
#     thetas : list
#         The coupling parameters for the p nodes.

#     Returns:
#     ---
#     unnormalized_likelihood : float
#         The unnormalized likelihood of the data parameterized by the coupling parameters.
#     '''
#     thetas_c = thetas[0]
#     thetas_s = thetas[1]
#     p = thetas_c.shape[0]
#     mus = np.arctan2(thetas_s, thetas_c)
#     kappas = np.sqrt(thetas_c**2 + thetas_s**2)

#     Y_diff = Y[:, :, np.newaxis] - Y[:, np.newaxis, :]
#     cos_terms = np.cos(Y_diff - mus)
#     unnormalized_likelihood = np.exp(np.sum(kappas * cos_terms))

#     return unnormalized_likelihood

# def generate_initial_gibbs_sample(mus,kappas):
#     #find all i,j such that kappas[i,j] != 0 in the upper triangular part of the matrix
#     kappa_triu = np.triu(kappas,1)
#     edges = np.argwhere(kappa_triu != 0)

#     #construct a linear system of equations such that j-node minus i-node is equal to mu[i,j]
#     #solve the system of equations
#     p = mus.shape[0]
#     n_edges = edges.shape[0]
#     A = np.zeros((n_edges,p))
#     b = np.zeros((n_edges,))
#     for idx,edge in enumerate(edges):
#         i,j = edge
#         A[idx,i] = -1
#         A[idx,j] = 1
#         b[idx] = mus[i,j]
#     try:
#         test_samp = np.linalg.solve(A,b)
#     except:
#         print('Singular matrix')
#         #least squares solution with minimum norm
#         test_samp = np.linalg.inv((A.T @ A + 0*np.eye(p))) @ A.T @ b
#         test_samp = np.angle(np.exp(1j*test_samp))
#         def objective(x):
#             return -np.exp(np.sum([kappas[i,j]*(np.cos(x[j] - x[i] - mus[i,j])) for i,j in edges]))
#         from scipy.optimize import minimize
#         res = minimize(objective,test_samp,method='L-BFGS-B')
#         test_samp = res.x

#     # test_samp = np.angle(np.exp(1j*test_samp))
#     print(test_samp)
#     return test_samp


    # MI = computeMutualInformation(kappas)
    # T = chowLiuTree(MI)
    # A = nx.adjacency_matrix(T).todense()

    # #use networkx to perform a breadth first search over the tree
    # root = 0
    # bfs = list(nx.bfs_edges(T,root))
    # test_samp = np.zeros((mus.shape[0],))
    # zero = np.random.uniform(-np.pi,np.pi)
    # for edge in bfs:
    #     i,j = edge
    #     if i == root:
    #         test_samp[j] = np.random.vonmises(mus[root,j],kappas[root,j]) + zero #mus[root,j]
    #     else:
    #         test_samp[j] = np.random.vonmises(test_samp[i] + mus[i,j],kappas[i,j])
    #         # if i < j:
    #         #     test_samp[j] = np.random.vonmises(test_samp[i] + mus[i,j],kappas[i,j]) #test_samp[i] + mus[i,j]
    #         # else:
    #         #     test_samp[j] = np.random.vonmises(test_samp[i] + mus[i,j],kappas[i,j])#test_samp[i] + mus[j,i]
    # test_samp = np.angle(np.exp(1j*test_samp))
    # print(test_samp)
    # return test_samp




# def generate_initial_gibbs_sample(mus,kappas):
#     p = mus.shape[0]
#     test_samp = np.zeros((p,))
#     not_visted = np.ones((p,))
#     not_visted[0] = 0
#     for i in range(1,p):
#         if kappas[0,i] != 0:
#             test_samp[i] = mus[0,i]# np.random.vonmises(mus[0,i],kappas[0,i])
#             not_visted[i] = 0
#     #find a non-zero node in test_samp that is connected to a node that hasn't been visited
#     curr_node = 1
#     while np.any(not_visted):
#         if curr_node == p:
#             break
#         if not_visted[curr_node]:
#             for j in range(p):
#                 if (kappas[curr_node,j] != 0) and (not_visted[j] == 0):
#                     test_samp[curr_node] = test_samp[j]+mus[j,curr_node]#np.random.vonmises(test_samp[j]+mus[j,curr_node],kappas[j,curr_node])
#                     not_visted[curr_node] = 0
#                     print(not_visted)
#                     break
#             curr_node += 1
#         else:
#             curr_node += 1

#     #if there are still nodes that haven't been visited, sample from the prior
#     if np.any(not_visted):
#         for i in range(p):
#             if not_visted[i]:
#                 test_samp[i] = np.random.uniform(-np.pi,np.pi)
#     test_samp = np.angle(np.exp(1j*test_samp))
#     return test_samp



# ## The following code is for the case where the active set for cosine and sine is different for each node

# def iso_sparse(Y,active_c,active_s, u = None,beta = None,alpha = None):
#     '''
#     The interaction screening objective for a generalized Potts model.
#     Only optimize over the active set of nodes.

#     Parameters
#     ----------
#     Y : np.array
#         The data matrix of size n x p of angles. p is the number of nodes and n is the number of samples.
#     active_c : list
#         The list of active cosine nodes to optimize over.
#     active_s : list
#         The list of active sine nodes to optimize over.
#     u : int
#         The node whose interactions are to be estimated
#     beta : float
#         The maximum coupling parameter. If None, beta is infinity.
#     alpha : float
#         The minimum coupling parameter. If None, alpha is set to 0.
    
#     Returns
#     -------
#     thetas_u_opt : np.array
#         The optimal coupling parameters for the u-th node.
#     '''

#     n,p = Y.shape

#     # Define the optimization variables
#     d_c = len(active_c)
#     d_s = len(active_s)
#     q = d_c + d_s
#     if q == 0:
#         return np.zeros(0)

#     thetas_u = cp.Variable(q)

#     # Define the constraints
#     constraints = []
#     if beta is not None:
#         constraints.append(cp.abs(thetas_u) <= beta)
    
#     # Define the objective
#     Y_offset_c = np.zeros((n,d_c))
#     Y_offset_s = np.zeros((n,d_s))
#     Y_u = Y[:,u]

#     for i in range(d_c):
#         if active_c[i] < u:
#             Y_offset_c[:,i] = Y_u - Y[:,active_c[i]]
#         elif active_c[i] > u:
#             Y_offset_c[:,i] = Y[:,active_c[i]] - Y_u

#     for i in range(d_s):
#         if active_s[i] < u:
#             Y_offset_s[:,i] = Y_u - Y[:,active_s[i]]
#         elif active_s[i] > u:
#             Y_offset_s[:,i] = Y[:,active_s[i]] - Y_u
    
#     cos_Y = np.cos(Y_offset_c)
#     sin_Y = np.sin(Y_offset_s)
#     Z = np.hstack([cos_Y,sin_Y])

#     # Generate the objective
#     Sn = 1/n * cp.sum(cp.exp(-Z @ thetas_u))
#     prob = cp.Problem(cp.Minimize(Sn))

#     try:
#         prob.solve(solver=cp.MOSEK)
#     except:
#         print('Mosek failed')
#         prob.solve(solver=cp.SCS)

#     thetas_u_opt = thetas_u.value
#     return thetas_u_opt


# def fit_sparse(Y,beta= None,alpha = None,lam = None):
#     '''
#     Fit the generalized Potts model to the data. Solve for parameters using interaction screening.
#     First solves the problem with a large penalty parameter and then solves the problem with no penalty.

#     Parameters
#     ----------
#     Y : np.array
#         The data matrix of size n x p of angles. p is the number of nodes and n is the number of samples.
#     beta : float
#         The maximum coupling parameter. If None, beta is infinity.
#         alpha : float
#         The minimum coupling parameter. If None, alpha is set to 0.
#     lam : float
#         The penalty parameter. If None, lam is set to default.

#     Returns
#     -------
#     thetas : list
#         The optimal coupling parameters for the p nodes.
#         thetas[0] are the coupling parameters for the cosine terms and thetas[1] are the coupling parameters for the sine terms.
#     '''

#     thetas_c_reg,thetas_s_reg = fit(Y,beta,alpha,lam)

#     n,p = Y.shape
#     theta_c_hat = np.zeros((p,p))
#     theta_s_hat = np.zeros((p,p))

#     thetas_dict = {}

#     active_sets_c = []
#     active_sets_s = []

#     #find the nodes with non-zero coupling parameters
#     for i in range(p):
#         active_set_c = np.where(thetas_c_reg[i] > 0)[0]
#         active_set_s = np.where(thetas_s_reg[i] > 0)[0]
#         active_sets_c.append(active_set_c)
#         active_sets_s.append(active_set_s)
#     print('Fitting sparse model')
#     print(active_sets_c)
#     print(active_sets_s)
#     results = Parallel(n_jobs=cpu_count())(delayed(iso_sparse)(Y,active_sets_c[u],active_sets_s[u],u,beta,alpha) for u in range(p))
#     for u in range(p):
#         thetas_dict[u] = results[u]

#     for u in range(p):
#         #fill in upper triangular part of theta_c_hat and theta_s_hat
#         #theta_c_hat is symmetric and theta_s_hat is skew-symmetric
#         d_c = len(active_sets_c[u])
#         d_s = len(active_sets_s[u])
#         for idx,i in enumerate(active_sets_c[u]):
#             if i < u:
#                 theta_c_hat[i,u] += thetas_dict[u][idx]
#             elif i > u:
#                 theta_c_hat[u,i] += thetas_dict[u][idx]

#         for idx,i in enumerate(active_sets_s[u]):
#             if i < u:
#                 theta_s_hat[i,u] += thetas_dict[u][idx+d_c]
#             elif i > u:
#                 theta_s_hat[u,i] += thetas_dict[u][idx+d_c]
        
#     theta_c_hat = theta_c_hat/2
#     theta_s_hat = theta_s_hat/2
#     theta_c_hat = theta_c_hat + theta_c_hat.T
#     theta_s_hat = theta_s_hat - theta_s_hat.T

#     thetas = [theta_c_hat,theta_s_hat]

#     return thetas