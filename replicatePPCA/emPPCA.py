import numpy as np
from numpy import shape, isnan, nanmean, mean, zeros, log, cov, pi
from numpy.random import normal
from numpy.linalg import inv, det, eig, norm
from numpy import trace as tr
from scipy.linalg import orth
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.neighbors import KNeighborsClassifier as KNC

# progress bar
try:
    get_ipython
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm


    
class emPPCA:
        
    # ------------------------------------------
    # Initialisation 
    # ------------------------------------------
    
    def __init__(self, Y, q, textlabel=None, label=None):
        """
        Initialize the model
        
        Parameters
        ----------
        Y : List[np.ndarray]
            Data list for the R replicates, each replicate is a matrix with each observation in a row
        q : int
            Number of retained dimensions
        textlabel : List[List]
                    The text labels or anything original to represent the classes (or not)
        label : List[List]
                Numerical labels of observations in each replicate, from 0 to the number of classes minue one
        """
        # Essential parameters
        self.Y = Y
        self.q = q
        self.p = self.Y[0].shape[1] # Number of features should match
        self.R = len(self.Y)
        self.n = [rep.shape[0] for rep in self.Y]
        self.textlabel = textlabel
        self.label = label
        self.objective = [np.NINF]
        
        # Output parameters
        self.eigvals = None # Eigenvalues for true PCs
        self.X = None # Low-dimensional representations
        self.W = None # Corrected principal components 
        self.recon = None
        self.sigma = None
        self.Y_imp = None
        
        # Dealing with missing entries
        self.missings = [isnan(rep) for rep in Y] # Missing locations
        self.n_nan = [missing.sum() for missing in self.missings] # Number of missing values
        self.E_Y = [rep - nanmean(rep, axis=0) for rep in self.Y] # 
        for r, missing in enumerate(self.missings):
            self.E_Y[r][missing] = 0
            
        # Get explicit class labels in texts and colors (without repetition)
        if (self.textlabel is not None) & (self.label is not None):
            self.text_classes = list(set([item for sublists in self.textlabel for item in sublists]))
            self.color_classes = list(set([item for sublists in self.label for item in sublists]))
        else:
            self.text_classes = None
            self.color_classes = None
        
        
    # ------------------------------------------
    # EM algorithm
    # ------------------------------------------
    
    def fit(self, tol=1e-6, max_iter=1000, seed=None):
        """
        Run EM algorithm for PPCA, on replicated data of possibly different sizes or with missing values 

        Parameters
        ----------
        param Y:   Data matrix of n*p, with each row an observation, type NumPy array
        param q:   Dimension of latent variables x
        return:    Lower dimensional representations 'X', Principal components 'W_pc', Isotropic variance 'sigma', 
                   Observation matrix with imputation 'Y_imp', Lower rank reconstruction 'recon'
        """
        
        # Initialization
        if seed is not None:
            np.random.seed(seed)
        self.W = normal(loc=0, scale=1, size=(self.p, self.q))
        self.E_X = [rep @ self.W @ inv(self.W.T @ self.W) for rep in self.E_Y]
        self.recon = [rep @ self.W.T for rep in self.E_X]
        for i, missing in enumerate(self.missings):
            self.recon[i][missing] = 0
        self.sigma = [np.sum((self.recon[i] - self.E_Y[i])**2) / (self.n[i] * self.p - self.n_nan[i]) for i in range(self.R)]

        # EM Iterations
        epoch = 1
        while epoch < max_iter:

            # E-step
            M_inv = [inv(self.W.T @ self.W + sigma * np.eye(self.q)) for sigma in self.sigma]
            for i in range(self.R):
                if self.n_nan[i] > 0:
                    self.E_Y[i][self.missings[i]] = (self.E_X[i] @ self.W.T)[self.missings[i]]
            self.E_X = [self.E_Y[i] @ self.W @ M_inv[i] for i in range(self.R)]

            # M-step for sigma (step1)
            sigma_new = [
                1 / (self.n[i] * self.p) * (self.n[i] * self.sigma[i] * tr(self.W @ M_inv[i] @ self.W.T) + 
                np.sum((self.E_X[i] @ self.W.T - self.E_Y[i])**2) + self.n_nan[i] * self.sigma[i]) 
                for i in range(self.R)
            ]

            # M-step for W (step2)
            weighted_1, weighted_2 = np.zeros((self.p, self.q)), np.zeros((self.q,self.q))
            for i in range(self.R):
                weighted_1 += (self.E_Y[i].T @ self.E_X[i]) / sigma_new[i]
                weighted_2 += (self.n[i]*self.sigma[i] * M_inv[i] + self.E_X[i].T @ self.E_X[i]) / sigma_new[i]
            W_new = weighted_1 @ inv(weighted_2)
            
            # Convergence check
            self.log_likelihood(sigma_new, M_inv)
            if epoch > 1:
                change = abs((self.objective[epoch-1] - self.objective[epoch]) / self.objective[epoch-1]) # Relative Change
            else:
                change = tol
            if change < tol:
                print(f'EM algorithm converged with {epoch} iterations; with relative change {change}.')
                break
            else:
                self.sigma = sigma_new
                self.W = W_new
                epoch = epoch + 1

        # Principal subspace has been found, true PCs remain unknown
        self.true_pc()

        # Other outputs
        self.Y_imp = [self.E_Y[r] + nanmean(self.Y[r], axis=0) for r in range(self.R)]
        self.recon = [rep @ self.W.T for rep in self.X]

        
    def log_likelihood(self, sigma_new, M_inv):
        """
        Check convergence & objective function \exist a techique for comp-cost at high dim\
        """
        obj = 0
        for r in range(self.R):
            obj += - self.n[r] * self.p * log(2 * pi * sigma_new[r]) / 2
            obj += - tr(self.E_X[r] @ self.E_X[r].T) / 2
            obj += - self.n[r] * (self.sigma[r] * tr(M_inv[r]) - log(det(self.sigma[r] * M_inv[r]))) / 2
            obj += self.n_nan[r] * log(self.sigma[r]) / 2
        self.objective.append(obj)
        
        
    def true_pc(self):
        """
        Recover true principal components from the principal subspace
        """
        data = np.concatenate(self.E_Y, axis=0)

        basis = orth(self.W) # Orthonormal basis of the principal subspace
        data_p = data @ basis # Projected observation matrix on the principal subspace
        S_p = cov(data_p.T) # Projected covariance matrix
        vals, vecs = eig(S_p) 
        idx = list(reversed(np.argsort(vals))) # Sort in decreasing order
        self.vals = vals[idx]; vecs = vecs[:,idx]
        self.X = [self.E_Y[i] @ basis @ vecs for i in range(self.R)] 
        self.W = basis @ vecs # Corrected principal components

    
    # ------------------------------------------
    # Convergence plot
    # ------------------------------------------

    def ll_plot(self, name=None):
        """
        Plot log-likelihood against number of iterations
        """
        epoch = len(self.objective)
        plt.figure(figsize=(6,4))
        plt.plot(self.objective)
        plt.title('The Log-Likelihood over iterations (after each M-step)')
        plt.xlabel('Iteration')
        plt.ylabel('Log-likelihood')
        if name is not None:
            plt.savefig(f'{name}.pdf')
        plt.show()


    # ------------------------------------------
    # Variance plot
    # ------------------------------------------
    
    def var_plot(self, name=None):
        """
        Variance unexplained proportion across replicates
        """
        fig, ax = plt.subplots(1,2, figsize=(16,6), gridspec_kw={'width_ratios': [7,1]})
        # Variance line plot
        ax[0].plot(sorted(self.sigma),'-o',markersize=4)
        ax[0].set_xticks(np.arange(self.R), np.argsort(self.sigma)+1);
        ax[0].set_xlabel('Replicates')
        ax[0].set_ylabel('Variance unexplained')
        # Variance box plot
        ax[1].boxplot(self.sigma, notch=True)
        plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        fig.suptitle('Proportion of variance unexplained across the replicates')
        fig.tight_layout()
        if name is not None:
            plt.savefig(f'{name}.pdf')
        plt.show()
        
        
    # ------------------------------------------
    # Low-dimensional representation plot
    # ------------------------------------------
    
    def nnline(self, X_tot, ax):
        # Find nearest neighbors
        nbrs = NN(n_neighbors=2, algorithm='ball_tree').fit(X_tot[:,0:2])
        distances, indices = nbrs.kneighbors(X_tot[:,0:2])
        # Number labels
        #label = [label_rep[i] for i in range(R)]
        cla = np.array([item for sublist in self.label for item in sublist])
        
        # 1-NN fit
        knn = KNC(n_neighbors=1)
        knn.fit(X_tot[:,0:2], cla)
        # Create a mesh grid for the contour plot
        x_min, x_max = X_tot[:, 0].min() - 1, X_tot[:, 0].max() + 1
        y_min, y_max = X_tot[:, 1].min() - 1, X_tot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        # Predict the labels for the mesh grid
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour of the decision boundary
        ax.contour(xx, yy, Z)
        
        
    def X_plot(self, pcs=[0,1], reps=None, compare=None, NN=True, lims=None, separate_legend=False, name=None):
        """
        Scatter plot of low-dimensional representations, with optional 1-NN line fitted under the full dataset
        
        Parameters
        ----------
        pcs : List
              The PCs for visualization, list of two numbers
        reps : List
               The id of replicates, from 1 to R
        rep_compare : Boolean
                      If True, use color to indicate replicates; If False, to indicate classes (only if specified)
        """
        if reps is None:
            reps = [r for r in range(self.R)]
        pc1 = pcs[0]
        pc2 = pcs[1]
        fig, ax = plt.subplots(figsize=(8,5))
        col = 0
        
        # Case 1, distinguish replicates
        if compare=='replicate':
            for r in reps:
                if len(reps) <= 20:
                    cmap = plt.get_cmap('tab20')
                    colors = cmap(range(len(reps)))
                else:
                    cmap = plt.get_cmap('rainbow')
                    colors = cmap(np.linspace(0, 1, len(reps)))
                ax.scatter(self.X[r-1][:,pc1], self.X[r-1][:,pc2], c=np.stack([colors[col] for i in range(self.n[r-1])]),
                            alpha=0.75, s=20, label=f'Replicate {r}')
                if separate_legend==True:
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                else:
                    ax.legend()
                col += 1
                
        # Case 2, distinguish classes
        elif (compare=='class') & (self.color_classes is not None) & (self.text_classes is not None):
            if len(self.color_classes) <= 20:
                cmap = plt.get_cmap('tab20')
                colors = cmap(range(len(self.color_classes)))
            else:
                cmap = plt.get_cmap('rainbow')
                colors = cmap(np.linspace(0, 1, len(self.color_classes)))
            x = [[] for i in range(len(self.color_classes))]
            y = [[] for i in range(len(self.color_classes))]
            for i in range(len(self.color_classes)):
                for r in reps:
                    for j in range(self.n[r-1]):
                        if self.label[r-1][j]==i:
                            x[i].append(self.X[r-1][j,pc1])
                            y[i].append(self.X[r-1][j,pc2])
                ax.scatter(x[i], y[i], c=[colors[i]] * len(x[i]), alpha=0.75, s=20, label=self.text_classes[i])
            if separate_legend==True:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                ax.legend()

        # Case 3 (no color labeling)
        else:
            for r in reps:
                ax.scatter(self.X[r-1][:,pc1], self.X[r-1][:,pc2], c='#1f77b4', alpha=0.75, s=20)
              
        if (NN==True) & (self.color_classes is not None) & (self.text_classes is not None):
            self.nnline(np.concatenate(self.X), ax)
        if lims is not None:
            ax.axis(lims)
        if name is not None:
            plt.savefig(f'{name}.pdf')
        plt.show()
        
    
    # ------------------------------------------
    # Reconstruction certainty plot
    # ------------------------------------------
    
    def recon_uncertainty(self, name=None):
        distance = [[] for r in range(self.R)]
        d = np.linspace(0,10,100)
        prop_within = [[] for r in range(self.R)]
        
        # Distances calculation
        for r in range(self.R):
            for i in range(len(self.recon[r])):
                distance[r].append(np.linalg.norm(self.Y[r][i] - self.recon[r][i]))
                
        # Compare with standard error
        for r in range(self.R):
            for j in d:
                prop_within[r].append(len([i for i in distance[r] if i < j * np.sqrt(self.sigma[r])]) / self.n[r])
                
        # Visualization
        fig, ax = plt.subplots(figsize=(10,6))
        for r in range(self.R):
            plt.plot(d, prop_within[r], label=f'Replicate {r+1}')
        plt.axvline(x=2, color='r', linestyle='--',  linewidth=0.75)
        plt.axhline(y=0.95, color='r',  linewidth=0.75)
        
        # Legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        if name is not None:
            plt.savefig(f'{name}.pdf')
        plt.show()