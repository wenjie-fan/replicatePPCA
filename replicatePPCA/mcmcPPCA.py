import numpy as np
#import pandas as pd
#import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy import log, pi, cov
from numpy import trace as tr
from numpy.linalg import inv, det, eig
from scipy.stats import invgamma, multivariate_normal, norm, gaussian_kde
from scipy.special import gamma
from scipy.linalg import orth
from sklearn.metrics import adjusted_rand_score
import random
import os
import pickle
from numpy.random import normal

# progress bar
try:
    get_ipython
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm




class mcmcPPCA:
    
    # ------------------------------------------
    # Initialisation 
    # ------------------------------------------
    
    def __init__(self, Y, q, W=None, sigma=None, X=None):
        """
        Initializes the model.

        Parameters
        ----------
        Y : List[np.ndarray]]
            Data in replicate lists
        q : int
            Number of retained dimensions
        iterations : int
            Number of MCMC iterations (default is 50000)
        W : List[np.ndarray]
            User-defined initializations of prejection matrices W for all replicates (default N(0,1) for each entry)
        sigma : np.ndarray
            User-defined initializations of variances unexplained for all replicates (default based on W)
        X : np.ndarray
            User-defined initializations for low-dimensional representations for all replicates (default based on W)
            
        Notes
        -----
        1. Input Y should be standardised
        """
        # Parameters
        self.Y = Y
        self.n = [rep.shape[0] for rep in self.Y]
        self.p = self.Y[0].shape[1]
        self.q = q
        self.R = len(self.Y)
        self.iterations = 50000
        # Fitted parameters
        self.W_rec = None
        self.sigma_rec = None
        self.sigma_rej = None
        self.X_rec = None
        self.objective = None
        # Hyper parameters
        self.proposal_W = 0.015 # Proposal std 
        self.proposal_sigma = [51, 50] # Proposal [shape, scale/sigma]
        self.proposal_X = 0.055 # Proposal std
        self.prior_W = 1 # Prior std
        self.prior_sigma = [3, 1] # [shape, scale]
        self.prior_X = 1 # Prior std
        # Post-processings
        self.N_eff = None
        self.W_pc = None
        self.W_mean = None
        self.X_iterwise = None
        self.X = None
        self.X_mean = None
        self.eigvals = None
        self.X_pooled = None
        self.X_mean_pooled = None
        
        # MCMC initialization
        # Case 1, Automatically initialize
        if (W is None) and (sigma is None) and (X is None):
            self.W = normal(loc=0, scale=1, size=(self.p, self.q))
            self.X = [rep @ self.W @ inv(self.W.T @ self.W) for rep in self.Y] # Collect in rows each x_i, size n*q
            recon = [rep @ self.W.T for rep in self.X]
            self.sigma = [np.sum((recon[r]-self.Y[r])**2) / (self.n[r] * self.p) for r in range(self.R)]
            print(f"Initialised with sigmas: {self.sigma}")
            
        # Case 2, With user-defined initial W matrix
        elif (W is not None) and ((sigma is None) or (X is None)):
            self.W = W
            self.X = [rep @ self.W @ inv(self.W.T @ self.W) for rep in self.Y] # Collect in rows each x_i, size n*q
            recon = [rep @ self.W.T for rep in self.X]
            self.sigma = [np.sum((recon[r]-self.Y[r])**2) / (self.n[r] * self.p) for r in range(self.R)]
            print(f"Initialised with sigmas: {self.sigma}")
            
        # Case 2, Fully specified, with user-defined initial W, sigma, and X
        else: 
            self.W = W
            self.sigma = sigma
            self.X = X
            
        # Figure and results output folder
        self.fig_folder = 'mcmc_output_figures'
        self.result_folder = 'mcmc_output_results'
        if not os.path.exists(self.fig_folder):
            os.makedirs(self.fig_folder)
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
         
        
    def set_proposals(self, W, sigma, X):    
        self.proposal_W = W # Proposal std 
        self.proposal_sigma = sigma # Proposal [shape, scale/sigma]
        self.proposal_X = X # Proposal std
        
    def set_priors(self, W, sigma, X):
        self.prior_W = W # Prior std
        self.prior_sigma = sigma # [shape, scale]
        self.prior_X = X # Prior std
        
            
    # ------------------------------------------
    # Fit fully Bayesian PPCA
    # ------------------------------------------   
    
    def fit(self, iterations=50000):
        """
        Run MH_PPCA function
        """
        self.iterations = iterations
        self.MH_PPCA(self.Y, self.W, self.sigma, self.X) 

        
    def MH_PPCA(self, Y, W, sigma, X):
        """
        Metropolis-Hastings MCMC for Fully Bayesian PPCA with R replicates of different observations.

        Returns
        -------
        W_rec : np.ndarray
            (iterations,p,q) Array of W over time, full, shape [R * (iterations * p * q)].
        sigma_rec : np.ndarray
            Sigma over time, full, shape [R * (iterations)].
        sigma_rej : np.ndarray
            Rejected sigma over time, contains 0, shape [R * (iterations)].
        X_rec : np.ndarray
            (iterations,n,q) Array of X over time, full, shape [R * (iterations * n[r] * q)].

        Notes
        -----
        W, sigma, X record the latest states of parameters.
        """
        # W records
        W_rec = np.zeros((self.iterations,self.p,self.q))
        W_acc_rate = 0

        # Replicate-wise records
        sigma_rec = [np.zeros(self.iterations) for r in range(self.R)]
        sigma_rej = [np.zeros(self.iterations) for r in range(self.R)]
        X_rec = [np.zeros((self.iterations,self.n[r],self.q)) for r in range(self.R)]
        acc_rates = [np.zeros(2) for r in range(self.R)]

        # Objective records
        objective = []

        # MCMC iterations
        for i in tqdm(range(self.iterations)):

            # Propose W
            W_new = self.propose_W(W)
            log_term = self.log_likelihood(W,sigma,X,Y) + self.log_prior(W, sigma, X) + self.prob_W(W,W_new)
            log_term_new = self.log_likelihood(W_new,sigma,X,Y) + self.log_prior(W_new, sigma, X) + self.prob_W(W_new,W)
            if self.accept(log_term, log_term_new):
                W = W_new
                W_acc_rate += 1 / self.iterations
            W_rec[i] = W # If not accept previous state will be recorded 

            # Propose sigma
            for r in range(self.R):
                sigma_new = self.propose_sigma(sigma[r])
                log_term = self.repwise_ll( W, sigma[r], X[r], Y[r] ) \
                            + self.repwise_prior( W, sigma[r], X[r] ) \
                            + self.prob_sigma( sigma[r], sigma_new )
                log_term_new = self.repwise_ll( W, sigma_new, X[r], Y[r] ) \
                                + self.repwise_prior( W, sigma_new, X[r] ) \
                                + self.prob_sigma( sigma_new, sigma[r] )
                if self.accept(log_term, log_term_new):
                    sigma[r] = sigma_new
                    acc_rates[r][0] += 1 / self.iterations
                else: 
                    sigma_rej[r][i] = sigma_new
                sigma_rec[r][i] = sigma[r]

            # Propose X
            for r in range(self.R):
                X_new = self.propose_X(X[r])
                log_term = self.repwise_ll( W, sigma[r], X[r], Y[r] ) \
                            + self.repwise_prior( W, sigma[r], X[r] ) \
                            + self.prob_X( X[r], X_new )
                log_term_new = self.repwise_ll( W, sigma[r], X_new, Y[r] ) \
                                + self.repwise_prior( W, sigma[r], X_new ) \
                                + self.prob_X( X_new, X[r] )
                if self.accept(log_term, log_term_new):
                    X[r] = X_new
                    acc_rates[r][1] += 1 / self.iterations
                X_rec[r][i] = X[r]

            # Acceptance rate inspection
            if (i + 1) % int(self.iterations/10) == 0:
                r_scale = self.iterations / i
                tqdm.write(
                    f"Current acceptance rates: W, {W_acc_rate*r_scale}; "
                    f"sigma, {np.mean(acc_rates,0)[0]*r_scale}; "
                    f"X, {np.mean(acc_rates,0)[1]*r_scale}"
                )

            # Record log-likelihod
            objective.append(self.log_likelihood(W,sigma,X,Y))
        
        # Return values
        self.W_rec = W_rec
        self.sigma_rec = sigma_rec
        self.sigma_rej = sigma_rej
        self.X_rec = X_rec
        self.objective = objective
        
        
    # ------------------------------------------
    # Accompanying functions for MH algorithm
    # ------------------------------------------   
    
    def accept(self, x, x_new):
        """
        Decision function of acceptance or rejection in MH algorithm
        """
        u = np.random.uniform(0,1)
        if (x_new - x) > np.log(u):
            return True
        else:
            return False
        
    # -------- Proposals --------

    def propose_W(self, W):
        return W + normal(0, self.proposal_W, (self.p,self.q))

    def propose_sigma(self, sigma):
        return invgamma.rvs(self.proposal_sigma[0], scale = self.proposal_sigma[1] * sigma)

    def propose_X(self, X):
        n = X.shape[0]
        return X + normal(0, self.proposal_X, (n,self.q))
        
    # -------- Transition probabilities (x -> x_new) --------

    def prob_W(self, x, x_new):
        trans_prob = 1
        for i in range(self.q):
            trans_prob *= multivariate_normal.pdf(x_new[:,i] - x[:,i], mean = np.zeros(self.p), cov = self.proposal_W * np.eye(self.p))
        return log(trans_prob)

    def prob_sigma(self, x, x_new):
        return log(invgamma.pdf(x_new, self.proposal_sigma[0], scale = self.proposal_sigma[1] * x)) 

    def prob_X(self, x, x_new):
        trans_prob = 1
        n = x.shape[0]
        for i in range(n):
            trans_prob *= multivariate_normal.pdf(x_new[i,:] - x[i,:], mean = np.zeros(self.q), cov = self.proposal_X * np.eye(self.q))
        return log(trans_prob)
    
    # -------- Priors --------
    
    def log_prior(self, W, sigma, X):
        """
        Log-priors, for MH ratio of W
        """
        # Log prior computation
        lp = - self.p * self.q * log(2 * pi * self.prior_W) / 2 - np.sum(W**2) / (2 * self.prior_W) # W
        for r in range(self.R):
            lp += log(invgamma.pdf(sigma[r], self.prior_sigma[0], scale = self.prior_sigma[1])) # sigma
            lp += - self.n[r] * self.q * log(2 * pi * self.prior_X) / 2 - np.sum(X[r]**2) / (2 * self.prior_X) # X
        return lp

    def repwise_prior(self, W, sigma, X):
        """
        Replicate-wise log-priors, for MH ratio of sigma and X
        """
        # Log prior computation
        n = X.shape[0]
        lp = - self.p * self.q * log(2 * pi * self.prior_W) / 2 - np.sum(W**2) / (2 * self.prior_W) # W
        lp += log(invgamma.pdf(sigma, self.prior_sigma[0], scale = self.prior_sigma[1])) # sigma
        lp += - n * self.q * log(2 * pi * self.prior_X) / 2 - np.sum(X**2) / (2 * self.prior_X) # X
        return lp
    
    # -------- Likelihoods --------
    
    def log_likelihood(self, W, sigma, X, Y):
        """
        Log-likelihood, for MH ratio of W and inspection of MCMC equilibrium
        """
        ll = 0
        for r in range(self.R):
            ll += - self.n[r] * self.p * log(2 * pi * sigma[r]) / 2 - np.sum((Y[r] - X[r] @ W.T)**2) / (2 * sigma[r])
        return ll
    
    def repwise_ll(self, W, sigma, X, Y):
        """
        Replicate-wise log-likelihood, for MH ratio of sigma and X
        """
        n = Y.shape[0]
        return - n * self.p * log(2 * pi * sigma) / 2 - np.sum((Y - X @ W.T)**2) / (2 * sigma)

    
    # ------------------------------------------
    # Convergence plots
    # ------------------------------------------

    def plot_sigma(self, r):
        """
        Plot accepted and rejected samples of sigma
        
        Parameters
        -------
        r : int
            The replicate number for visualising, from 1 to R
        """
        r -= 1
        plt.figure(figsize=(10,5))
        rej_id = np.nonzero(self.sigma_rej[r])[0]
        plt.plot(rej_id, self.sigma_rej[r][rej_id],'#1B6B93',alpha=0.3,label='Rejected proposals')
        plt.plot(self.sigma_rec[r],'#1B6B93',alpha=1,label='Accepted proposals')
        plt.xlim([0, self.iterations])
        plt.legend()
        plt.xlabel('Number of iterations')
        plt.ylabel('Proposed variance values')
        plt.title('Convergence of variance unexplained (sigma)')
        plt.show()
    
    def plot_likelihood(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.objective,'#1B6B93',alpha=1)
        plt.xlabel('Number of iterations')
        plt.ylabel('Total log-likelihood of the iteration')
        plt.title('Convergence of the total log-likelihood')
        plt.show()
        
        
    # ------------------------------------------
    # Post-processing principal components
    # ------------------------------------------  
        
    def true_pc(self, W):
        """
        Recover true principal components, as the learned W only shows the principal subspace
        """
        data = np.concatenate(self.Y, axis=0)
        
        basis = orth(W) # Orthonormal basis of the principal subspace
        data_p = data @ basis # Projected observation matrix on the principal subspace
        S_p = cov(data_p.T) # Projected covariance matrix
        vals, vecs = eig(S_p) 
        idx = list(reversed(np.argsort(vals))) # Sort in decreasing order
        vals = vals[idx]; vecs = vecs[:,idx]
        
        W_pc = basis @ vecs # Corrected principal components
        X = [self.Y[i] @ W_pc for i in range(self.R)]

        return W_pc, X, vals
        
    
    def post_process(self, burn_in):
        """
        Post-processing the MCMC sample
        1. Burn-in cut-off
        2. Recover true PCs
        3. Match direction of PCs (issue caused by invariance of PC direction)
        4. Get mean PCs and its projections
        Finally return the processed variables
        
        Parameters
        -------
        burn_in : The cut-off points for the start of MCMC equilibrium, usually inspect from ".plot_likelihood"
        
        Notes
        -------
        1. True PCs and low-dimensional representations are recovered iteration-wise
        2. Mean PCs and Mean low-dimensional representations are therefore taken along the iterations
        """
        # Effective sample of W
        W_eff = self.W_rec[burn_in:]
        N_eff = W_eff.shape[0]
        # Initialization
        W_pc = np.zeros((N_eff,self.p,self.q)) # Shape (N_eff, p, q)
        eigvals = np.zeros((N_eff,self.q))
        X = []
        # Solve for true principal components
        for i, matrix in enumerate(W_eff):
            W_pc[i], projs, eigvals[i] = self.true_pc(matrix)
            X.append(projs)

        # Adjust the PCs to same signs (by checking the angles between)
        # At least first two PCs can be recognized
        for i in range(N_eff):
            for j in range(self.q):
                PC_ref = W_pc[0,:,j]; PC2 = W_pc[i,:,j] 
                cosine = np.dot(PC_ref, PC2) / (np.linalg.norm(PC_ref) * np.linalg.norm(PC2))
                if cosine < 0:
                    W_pc[i,:,j] = - W_pc[i,:,j]
                    for r in range(self.R):
                        X[i][r][:,j] = - X[i][r][:,j]

        # Mean of recovered true projection matrices & Switched-index X_repwise
        W_mean = np.mean(W_pc, axis=0) # Sample mean
        X_sum = [0 for i in range(self.R)]
        X_repwise = [[None for _ in range(N_eff)] for _ in range(self.R)]
        for t in range(N_eff):
            for r in range(self.R):
                X_repwise[r][t] = X[t][r] # Switch from iteration-wise [N_eff * [R * (n * q)]] to replicate-wise
        X_mean_repwise = [np.mean(X_repwise[r],axis=0) for r in range(self.R)] # Replicate-wise sample means [R * (n * q)]
        
        ### ---------- Revising --------------------------
        
        # Return computed values
        self.N_eff = N_eff                    # Number of effective samples
        self.W_rec = W_pc                      # True PCs (N_eff * p * q)
        self.W_mean = W_mean                  # Mean true PCs (p * q)
        self.X_rec = X                   # Record of projections [N_eff * [R * (n * q)]]
        #self.X_iterwise_mean = X_mean         # Iterwise mean projections 
        self.X_rec_repwise = X_repwise                    # Record of projections [R * [N_eff * (n * q)]]
        self.X_mean = X_mean_repwise          # Mean projections of replicates [R * (n * q)]    (*)
        self.eigvals_rec = eigvals                # Record of recovered eigenvalues (N_eff * q)
        
        # Pool data [N_eff * [R * (n[r] * q)]] to [N_eff * (sum(n) * q)]
        self.X_mean_pooled = np.concatenate(self.X_mean, axis=0) # Pooled mean projections (sum(n) * q)
        self.X_rec_pooled = [np.concatenate([sublist[i] for sublist in self.X_rec_repwise], axis=0) for i in range(self.N_eff)]
        self.X_rec_pooled = np.stack(self.X_rec_pooled, axis=0) # Pooled projections (N_eff * sum(n) * q)
        
        # self.X_rec_pooled = np.zeros((self.N_eff, sum(self.n), self.q))
        # for t in range(self.N_eff):
        #     n_indicator = 0
        #     for r in range(self.R):
        #         self.X_rec_pooled[t,n_indicator:(n_indicator+self.n[r]),:] = self.X_rec_repwise[r][t] 
        #         n_indicator += self.n[r]
        # self.X_mean_pooled = np.mean(self.X_rec_pooled, axis=0)
        
        
    # ------------------------------------------
    # Contour plots
    # ------------------------------------------  
        
    def contour_plot_repwise(self, a, b, colors, bw_method, levels, ax):
        # Iterate through observations
        for i in range(a.shape[1]): 
            # stack them into a 2D array
            data = np.vstack([a[:,i], b[:,i]])
            kde = gaussian_kde(data,bw_method=bw_method)
            
            # Use standard deviation to estimate the support of KDEs
            mean_a, mean_b = np.mean(a[:, i]), np.mean(b[:, i])
            std = max(np.std(a[:, i]), np.std(b[:, i]))
            x_min, x_max = mean_a - 10 * std, mean_a + 10 * std
            y_min, y_max = mean_b - 10 * std, mean_b + 10 * std

            # generate a grid on which to evaluate the kde
            xgrid = np.linspace(x_min, x_max, 200) 
            ygrid = np.linspace(y_min, y_max, 200)
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
            Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
            Z = Z.reshape(Xgrid.shape)
        
            # plot the contour line
            ax.contour(Xgrid, Ygrid, Z, levels=levels, colors=[colors[i]], alpha=0.8)
            
            
    def contour_plot(self, a, b, colors, bw_method, levels, ax):
        # Iterate through observations
        for i in range(a.shape[1]): 
            # stack them into a 2D array
            data = np.vstack([a[:,i], b[:,i]])
            kde = gaussian_kde(data,bw_method=bw_method)
            
            # Use standard deviation to estimate the support of KDEs
            mean_a, mean_b = np.mean(a[:, i]), np.mean(b[:, i])
            std = max(np.std(a[:, i]), np.std(b[:, i]))
            x_min, x_max = mean_a - 10 * std, mean_a + 10 * std
            y_min, y_max = mean_b - 10 * std, mean_b + 10 * std

            # generate a grid on which to evaluate the kde
            xgrid = np.linspace(x_min, x_max, 200) 
            ygrid = np.linspace(y_min, y_max, 200)
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
            Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
            Z = Z.reshape(Xgrid.shape)
        
            # plot the contour line
            ax.contour(Xgrid, Ygrid, Z, levels=levels, colors=colors, alpha=0.8)

            
    def X_contours(self, repwise=True, reps=None, reduct_proportion=0, bw_method=0.5, levels=[0.05,1,4], lims=None, savefig=None):
        """
        Visualise distribution of low-dimensional representations using contours
        
        Parameters
        -------
        repwise : boolean
                  Whether plot the contours one by one in replicates
        reduct_proportion : float
                            The proportion of MCMC samples to be reduced, for faster computation
        """
        if reps is None:
            reps = [r for r in range(self.R)]
        else:
            reps = [r-1 for r in reps]

        if repwise==True:
            # Reduct sample size by a specified proportion
            reducted = random.sample(range(self.N_eff), int(self.N_eff*(1-reduct_proportion)))
            X_reduct = [[self.X_rec_repwise[r][t] for t in reducted] for r in range(self.R)] # Reduct to [R * N_reduct * (n * q)]
            
            # Optimise grid layout
            grid_size = int(np.ceil(np.sqrt(len(reps))))
            num_rows = grid_size
            num_columns = grid_size
            
            # Create figure
            fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 12), squeeze=False)
            fig.tight_layout(pad=3.0)
            # Get figure limits
            x_minus = np.min(self.X_mean_pooled[:, 0]) - 1
            x_plus = np.max(self.X_mean_pooled[:, 0]) + 1
            y_minus = np.min(self.X_mean_pooled[:, 1]) - 1
            y_plus = np.max(self.X_mean_pooled[:, 1]) + 1
            
            # For each replicate in reps
            for i, r in enumerate(reps):
                # Get the axes object for current replicate
                ax = axes[i // num_columns, i % num_columns]
                # Get color maps for current axes
                if max(self.n) <= 20:
                    cmap = plt.get_cmap('tab20')
                    colors = cmap(range(self.n[r]))
                else:
                    cmap = plt.get_cmap('rainbow') # For less observations (<=20), "tab20" may be better
                    colors = cmap(np.linspace(0, 1, self.n[r]))
                # Contour plotting
                self.contour_plot_repwise(np.array(X_reduct[r])[:,:,0], np.array(X_reduct[r])[:,:,1], colors, bw_method, levels, ax)
                ax.scatter(self.X_mean[r][:,0], self.X_mean[r][:,1], s=100, marker='^', color=colors)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                if lims is not None:
                    ax.axis(lims)
                else:
                    ax.set_xlim(x_minus, x_plus)
                    ax.set_ylim(y_minus, y_plus)
            fig.suptitle('MCMC distribution visualisation, with replicates')
            
            # Delete redundant axes
            for i in range(len(reps), num_rows * num_columns):
                fig.delaxes(axes[i // num_columns, i % num_columns])
            plt.show()
        
        ### ---------------------------- X_reduct is not able to apply below ------------- revising
        
        else:
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            # Get color map
            if self.R <= 20:
                cmap = plt.get_cmap('tab20')
                colors = cmap(reps)
            else:
                cmap = plt.get_cmap('rainbow') # For less replicates (<=20), "tab20" may be better
                colors = cmap(np.linspace(0, 1, len(reps)))
            
            # For each replicate in reps
            n_indicator = 0
            for i, r in enumerate(reps):
                self.contour_plot(np.array(self.X_rec_repwise[r])[:,:,0], 
                                  np.array(self.X_rec_repwise[r])[:,:,1], 
                                  colors[i], bw_method, levels, ax
                )
                ax.scatter(self.X_mean[r][:,0], self.X_mean[r][:,1], s=100, marker='^', color=colors[i])
                # self.contour_plot(self.X_rec_pooled[:,n_indicator:(n_indicator+self.n[r]),0], 
                #                   self.X_rec_pooled[:,n_indicator:(n_indicator+self.n[r]),1], 
                #                   colors[i], bw_method, levels, ax
                # )
                # ax.scatter(self.X_mean_pooled[n_indicator:(n_indicator+self.n[r]),0], 
                #            self.X_mean_pooled[n_indicator:(n_indicator+self.n[r]),1], 
                #            s=100, marker='^', color=colors[i]
                # )
                # n_indicator += self.n[r]
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('MCMC distribution visualisation, with replicates')
            ax.legend([f'Replicate {r+1}' for r in reps])
            if savefig is not None:
                plt.savefig(f'{self.fig_folder}/{savefig}.pdf')
            if lims is not None:
                ax.axis(lims)
            plt.show()

    
    # ------------------------------------------
    # Save and load data
    # ------------------------------------------  
        
    def save(self, filename):
        with open(f'{self.result_folder}/{filename}', 'wb') as file:
            pickle.dump(self.__dict__, file)
    
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            attributes = pickle.load(file)
            instance = cls.__new__(cls)
            instance.__dict__.update(attributes)
            return instance
