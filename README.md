# The `replicatePPCA` package

The [**replicatePPCA**](https://github.com/wenjie-fan/replicatePPCA/) (Replicated-data Probabilistic Principal Component Analysis) package is a Python package for analyzing mass spectrometry (MS) data in proteomics. Serving as an extension to probabilistic principal component analysis (PPCA) [[1]](#1), the package provides model fitting, visualizations, replicate analysis, and clustering uncertainty quantification and beyond. These results can be further passed on to unsupervised clustering algorithms.

## Basic ideas and concepts

As extracted from the report abstract:

Principal component analysis (PCA) is frequently used for the visualization and unsupervised clustering of mass spectrometry (MS) proteomics data. However, there is no intuitive method to handle missing values, and the intricate structure of experimental design is often neglected. A typical example is data from biological experiments with replicate structure, which displays both variations and correlations among replicates, and suffers from incomplete observations. Probabilistic principal component analysis (PPCA) recovers missing values soundly within a probabilistic framework, yet it still overlooks the replicate-wise variations. In this study, we extend PPCA to be conscious of the replicated model for MS data with replicated structures and possibly broader. We present  two distinct inference approaches: one focuses on the likelihood model by the Expectation-Maximization (EM) algorithm, while the other lies in the Bayesian framework employing the Markov chain Monte Carlo (MCMC) method. Our findings indicate that both approaches yield more nuanced analyses, some aspects of which have previously been ignored by researchers. The likelihood-centric approach hints replicates that may be unsuitable for PCA and tackles the challenge of missing values. In contrast, the Bayesian approach provides insightful visualizations, quantifies clustering uncertainty, and determines the risk-minimizing clustering rule.

## User instruction

Users should first download the modules and place them in the same folder to import successfully. Modules can be imported as follows:

``` python
from replicatePPCA import emPPCA
```

``` python
from replicatePPCA import mcmcPPCA
```

## Vignettes and documentation

Currently, two [vignettes](https://github.com/wenjie-fan/replicatePPCA/tree/main/vignettes) are available: one for **`emPPCA`** and the other for **`mcmcPPCA`**. Both are presented in Python notebooks, with each showcasing application examples on [datasets](https://github.com/wenjie-fan/replicatePPCA/tree/main/datasets) from the following sources:

-   **Tobomovirus**: Data for 38 distinct tobamoviruses with features including the number of amino acid residues per molecule of coat protein, previously used on P.291 in [[2]](#2). The data is randomly added with varying degrees of noise to establish a replicated structure.

-   **pSCoPE** (Fig. 4a): Mass-spectrometry proteomics data under pSCoPE, comparing the bone marrow-derived macrophages (BMDM) that are treated or untreated with lipopolysaccharide (LPS) [[3]](#3). The experiments were executed in 40 batches; hence we broadly assume these to represent 40 replicates.

-   **plexDIA** (Fig. 6p): Mass-spectrometry proteomics data under plexDIA, comparing Melanoma, pancreatic ductal adenocarcinoma (PDAC), and monocyte (U-937) [[4]](#4). Two different mass spectrometer (QE and timsTOF) were used for data acquisition, which could be considered as two replicates.

Run Time Notes: The `mcmcPPCA` fitting function might require around half an hour to run on personal computers, subject to the MCMC sample size.

Each module also owns its [documentation](https://github.com/wenjie-fan/replicatePPCA/tree/main/documentation), offering an in-depth elucidation of available attributes and methods.

## References

<a id="1">[1]</a> Tipping, M. E., & Bishop, C. M. (1999). Probabilistic principal component analysis. Journal of the Royal Statistical Society Series B: Statistical Methodology, 61(3), 611-622.

<a id="1">[2]</a> Ripley, B. D. (2007). *Pattern recognition and neural networks*. Cambridge university press.

<a id="1">[3]</a> Huffman, R. G., Leduc, A., Wichmann, C., Di Gioia, M., Borriello, F., Specht, H., \... & Slavov, N. (2023). Prioritized mass spectrometry increases the depth, sensitivity and data completeness of single-cell proteomics. *Nature methods*, *20*(5), 714-722.

<a id="1">[4]</a> Derks, J., Leduc, A., Wallmann, G., Huffman, R. G., Willetts, M., Khan, S., \... & Slavov, N. (2023). Increasing the throughput of sensitive proteomics by plexDIA. *Nature biotechnology*, *41*(1), 50-59.
