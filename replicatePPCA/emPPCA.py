import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
from numpy import log, cov, pi, trace
from numpy.linalg import inv, det, eig, norm
from scipy.linalg import orth
from matplotlib.colors import Normalize
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
    
    def __init__(self, Y, q, label=None, markers=None, colors=None):
        """
        Initialize the model
        
        Parameters
        ----------
        Y : List[np.ndarray]
            Data list for the R replicates, each replicate is a matrix with each observation in a row
        q : int
            Number of retained dimensions
        label : List[List]
                The text labels or anything original to represent the classes of observations
        markers : List
                  User may specify the markers for different replicates
        colors : List
                 User may specify the colors for different classes
        """
        # Essential parameters
        self.Y = copy.deepcopy(Y)
        self.q = q
        self.p = self.Y[0].shape[1]                       # Number of features, the same across replicates
        self.R = len(self.Y)                              # Number of replicates
        self.n = [rep.shape[0] for rep in self.Y]         # Number of observations within each replicate, as a list
        self.objective = [np.NINF]                        # Records of the objective, as a list
        self.missings = [np.isnan(rep) for rep in self.Y] # Missing locations
        self.n_nan = [rep.sum() for rep in self.missings] # Counts of missing entries for the replicates
        
        # Text and its number labels, replicate markers
        self.textlabel = copy.deepcopy(label)  
        if label is not None:
            self.text_classes = list(set([item for sublists in self.textlabel for item in sublists])) # Text labels as set
            number_dict = {text: idx for idx, text in enumerate(self.text_classes)}
            self.numberlabel = [[number_dict[text] for text in self.textlabel[r]] for r in range(self.R)] 
        else:
            self.text_classes = None  
            self.numberlabel = None
        if markers == None:
            self.marker = ['o', 's', 'D', '^', '*', 'd', '+', 'x', '|', '_']
        if colors == None:
            self.color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Output parameters
        self.eigvals = None # Recovered eigenvalues for the true PCs
        self.X = None       # Low-dimensional representations
        self.W = None       # Principal components stored as a matrix (in columns)
        self.recon = None   # Reconstructions of rank q
        self.sigma = None   # List of variance unexplained
        self.Y_imp = None   # List of imputed observation matrices
        
        # Abundance z-score
        if any(np.any(array) for array in self.missings):
            abundance = [[np.isnan(obs).sum() / self.p for obs in self.Y[r]] for r in range(self.R)] # Abundance percentages
            abundance_mean = np.array([np.mean(abundance[r]) for r in range(self.R)])
            abundance_std = np.array([np.std(abundance[r]) for r in range(self.R)])
            self.z_score = [[(obs - abundance_mean[r])/(abundance_std[r] + 1e-6) for obs in abundance[r]] for r in range(self.R)]
        else:
            self.z_score = [np.zeros(self.n[r]) for r in range(self.R)] # If there are no missing entries, abundances are set to be zeros
        
        # Dealing with missing entries
        # Set missing entries to be zero, as the data has been standardized
        self.E_Y = copy.deepcopy(self.Y)  # [rep - np.nanmean(rep, axis=0) for rep in self.Y] 
        for r, missing in enumerate(self.missings):
            self.E_Y[r][missing] = 0
            
        # Figure output folder
        self.folder = 'em_output_figures'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        
        
    # ------------------------------------------
    # EM algorithm
    # ------------------------------------------
    
    def fit(self, tol=1e-6, max_iter=1000, seed=None, verbose=True):
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
        self.W = np.random.normal(loc=0, scale=1, size=(self.p, self.q))
        self.E_X = [rep @ self.W @ inv(self.W.T @ self.W) for rep in self.E_Y]
        self.recon = [rep @ self.W.T for rep in self.E_X]
        for i, missing in enumerate(self.missings):
            self.recon[i][missing] = 0
        self.sigma = [np.sum((self.recon[i] - self.E_Y[i])**2) / (self.n[i] * self.p - self.n_nan[i]) for i in range(self.R)]

        # EM Iterations
        epoch = 1
        while epoch < max_iter:

            # E-step
            self.M_inv = [inv(self.W.T @ self.W + sigma * np.eye(self.q)) for sigma in self.sigma]
            for i in range(self.R):
                if self.n_nan[i] > 0:
                    self.E_Y[i][self.missings[i]] = (self.E_X[i] @ self.W.T)[self.missings[i]]
            self.E_X = [self.E_Y[i] @ self.W @ self.M_inv[i] for i in range(self.R)]

            # M-step for sigma (step1)
            sigma_new = [
                1 / (self.n[i] * self.p) * (self.n[i] * self.sigma[i] * trace(self.W @ self.M_inv[i] @ self.W.T) + 
                np.sum((self.E_X[i] @ self.W.T - self.E_Y[i])**2) + self.n_nan[i] * self.sigma[i]) 
                for i in range(self.R)
            ]

            # M-step for W (step2)
            weighted_1, weighted_2 = np.zeros((self.p, self.q)), np.zeros((self.q,self.q))
            for i in range(self.R):
                weighted_1 += (self.E_Y[i].T @ self.E_X[i]) / sigma_new[i]
                weighted_2 += (self.n[i]*self.sigma[i] * self.M_inv[i] + self.E_X[i].T @ self.E_X[i]) / sigma_new[i]
            W_new = weighted_1 @ inv(weighted_2)
            
            # Convergence check
            self.log_likelihood(sigma_new, W_new)
            if epoch > 1:
                change = abs((self.objective[epoch-1] - self.objective[epoch]) / self.objective[epoch-1]) # Relative Change
            else:
                change = tol
            if change < tol:
                if verbose == True:
                    print(f'EM algorithm converged with {epoch} iterations; with relative change {change}.')
                break
            else:
                self.sigma = sigma_new
                self.W = W_new
                epoch = epoch + 1

        # Principal subspace has been found, true PCs remain unknown
        self.true_pc()

        # Other outputs
        self.Y_imp = [self.E_Y[r] for r in range(self.R)]
        self.recon = [rep @ self.W.T for rep in self.X]

        
    def log_likelihood(self, sigma_new, W_new):
        """
        Check convergence & objective function
        """
        obj = 0
        for r in range(self.R):
            obj += - self.n[r] * (self.p + self.q) * log(2 * pi) / 2
            obj += - self.n[r] * self.p * (1 + log(sigma_new[r])) / 2
            obj += - self.n[r] * (self.sigma[r] * trace(self.M_inv[r]) - log(self.sigma[r] * det(self.M_inv[r]))) / 2
            obj += - trace(self.E_X[r] @ self.E_X[r].T) / 2
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

    def likelihood_plot(self, name=None):
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
            plt.savefig(f'{self.folder}/{name}.pdf')
        plt.show()


    # ------------------------------------------
    # Variance plot
    # ------------------------------------------
    
    def variance_plot(self, name=None):
        """
        Variance unexplained proportion across replicates
        """
        fig, ax = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios': [7,1]})
        # Variance line plot
        ax[0].plot(sorted(self.sigma),'-o',markersize=4)
        ax[0].set_xticks(ticks=np.arange(self.R), minor=False)
        ax[0].set_xticklabels(np.argsort(self.sigma) + 1)
        ax[0].set_xlabel('Replicates')
        ax[0].set_ylabel('Variance unexplained')
        # Variance box plot
        ax[1].boxplot(self.sigma, notch=True)
        plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        fig.suptitle('Proportion of variance unexplained across the replicates')
        fig.tight_layout()
        if name is not None:
            plt.savefig(f'{self.folder}/{name}.pdf')
        plt.show()
        
        
    # ------------------------------------------
    # Low-dimensional representation plot
    # ------------------------------------------
    
    def nnline(self, X_tot, ax):
        # Find nearest neighbors
        nbrs = NN(n_neighbors=2, algorithm='ball_tree').fit(X_tot[:,0:2])
        distances, indices = nbrs.kneighbors(X_tot[:,0:2])
        # Number labels
        cla = np.array([item for sublist in self.numberlabel for item in sublist])
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
        
        
    def X_plot(self, pcs=[1,2], reps=None, compare=None, NN=True, lims=None, separate_legend=False, name=None):
        """
        Scatter plot of low-dimensional representations, with optional 1-NN line fitted under the full dataset
        
        Parameters
        ----------
        pcs : List
              The PCs for visualization, list of two distinct integers, each from 1 to q
        reps : List
               The id of replicates, from 1 to R
        rep_compare : Boolean
                      If True, use color to indicate replicates; If False, to indicate classes (only if specified)
        """
        if reps is None:
            reps = [r for r in range(self.R)]
        else:
            reps = [r-1 for r in reps]
        pc1 = pcs[0] - 1
        pc2 = pcs[1] - 1
        
        # Dataframe for scattering
        df = pd.DataFrame({'PC1': np.concatenate(self.X)[:,pc1], 
                           'PC2': np.concatenate(self.X)[:,pc2],
                           'rep': [i for i,count in enumerate(self.n) for _ in range(count)]})
        if self.textlabel is not None:
            df['class'] = np.concatenate(self.textlabel)   
        
        # Case 1, distinguish replicates
        if compare=='replicate':
            fig, ax = plt.subplots(figsize=(7,7))
            marker = 0
            for r in reps:
                ax.scatter(self.X[r][:,pc1], self.X[r][:,pc2], marker=self.marker[marker],
                            alpha=0.7, s=30, label=f'Replicate {r+1}')
                if separate_legend==True:
                    # box = ax.get_position()
                    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                else:
                    ax.legend()
                marker += 1   
                
        # Case 2, distinguish classes
        elif (compare=='class') & (self.text_classes is not None): 
            fig, ax = plt.subplots(figsize=(7,7))
            for text in self.text_classes:
                condition = (df['class'] == text)
                ax.scatter(df.loc[condition, 'PC1'], df.loc[condition, 'PC2'], alpha=0.7, s=30, label=text)
            if separate_legend==True:
                # box = ax.get_position()
                # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            else:
                ax.legend()                    
                
        # Case 3, distinguish simultaneously
        elif (compare==None) & (self.text_classes is not None):
            fig, ax = plt.subplots(figsize=(7,7))
            color = 0
            for text in self.text_classes:
                marker = 0
                for rep in reps:
                    condition = (df['class'] == text) & (df['rep'] == rep)
                    ax.scatter(df.loc[condition, 'PC1'], df.loc[condition, 'PC2'], 
                               marker=self.marker[marker], c=self.color[color], alpha=0.7, s=30, label=text)
                    marker += 1
                color += 1
            # Figure settings
            legend_labels = [mpatches.Patch(color=self.color[i], label=self.text_classes[i]) for i in range(color)]
            legend_labels += [plt.Line2D([0], [0], color='gray', marker=self.marker[r], linestyle='', label=f'Replicate {reps[r]+1}') for r in range(marker)]
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # ax.legend(handles=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Case 4, directly visualize
        else:
            fig, ax = plt.subplots(figsize=(7,7))
            ax.scatter(df['PC1'], df['PC2'], c='#1f77b4', alpha=0.7, s=30)
        
        # Others
        if (NN==True) & (self.text_classes is not None):
            self.nnline(np.concatenate(self.X), ax)
        if lims is not None:
            ax.axis(lims)
        if name is not None:
            plt.savefig(f'{self.folder}/{name}.pdf')
        plt.show()
        
        
    # ------------------------------------------
    # Feature abundance plot
    # ------------------------------------------
    def abundance_plot(self, rep=None, pcs=[1,2], lims=None, name=None):
        if rep is None:
            rep = [r for r in range(self.R)]
        else:
            rep = [r-1 for r in rep]
        if 0 not in self.n_nan:
            print('The dataset has full observations, hence the abundance plot is not representative.')
        pc1 = pcs[0] - 1
        pc2 = pcs[1] - 1
        z_scores = [score for r in rep for score in self.z_score[r]]
        X_flatten = np.concatenate([self.X[r] for r in rep], axis=0)
        # Scattering with Z-score visualization
        norm = Normalize(vmin=min(z_scores), vmax=max(z_scores))
        colors = plt.cm.coolwarm(norm(z_scores))
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(X_flatten[:,pc1], X_flatten[:,pc2], c=colors, alpha=0.75, s=20)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='coolwarm'), label='Abundance z score')
        if lims is not None:
            ax.axis(lims)
        if name is not None:
            plt.savefig(f'{self.folder}/{name}.pdf')
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
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        if name is not None:
            plt.savefig(f'{self.folder}/{name}.pdf')
        plt.show()
