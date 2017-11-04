# dev-meetup-dimensionality-reduction
Jupyter Notebooks to demonstrate Factor Analysis (FA), t-distributed stochastic neighbor embedding (t-SNE) and Auto Encoders (AE)

# Factor Analysis (FA)

The factor Analysis examples are a mix of the following three different blog posts:
1. [How to do factor analysis](https://blog.dominodatalab.com/how-to-do-factor-analysis) by [Nick Elprin](https://blog.dominodatalab.com/author/nick): this uses the standard R 'Psych' package.
1. [Fitting a Bayesian Factor Analysis Model in Stan](https://rfarouni.github.io/assets/projects/BayesianFactorAnalysis/BayesianFactorAnalysis.html) by [Rick Farouni](http://rfarouni.github.io): replicates the model in Stan.
1. [Probabilistic Factor Analysis Methods](https://www.cs.helsinki.fi/u/sakaya/tutorial) by [Suleiman A. Khan](https://users.ics.aalto.fi/suleiman/) & [Joseph H. Sakaya](https://www.cs.helsinki.fi/en/people/sakaya): uses the ADVI ([Automatic Variational Inference in Stan](https://arxiv.org/abs/1506.03431)) inference method rather than MCMC sampling and combines the model with ARD (Automatic Relevance Determination, see chapter 13.7 "Automatic relevance determination (ARD)/sparse Bayesian learning (SBL)" of [Machine Learning: A Probabilistic Perspective](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020)) to automatically detect the number of latent dimensions.

The below notebooks are translations of the above examples to python plus some additional features that I thought were missing from the above, e.g. an evaluation of the quality of the fit.
* [Factor Analysis via R 'Psych'](https://nbviewer.jupyter.org/github/cs224/dev-meetup-dimensionality-reduction/blob/master/factor-analysis.ipynb): Using the standard R 'Psych' package.
* [Factor Analysis via MCMC sampling](https://nbviewer.jupyter.org/github/cs224/dev-meetup-dimensionality-reduction/blob/master/factor-analysis-via-mcmc-sampling.ipynb): Reimplementing the model in [Stan](http://mc-stan.org/).
* [Factor Analysis with Automatic Relevance Determination and ADVI](https://nbviewer.jupyter.org/github/cs224/dev-meetup-dimensionality-reduction/blob/master/factor-analysis-with-automatic-relevance-determination-and-advi.ipynb): Adding Automatic Relevance Determination (ARD) to the model.
* Some geometric aspects of the multi-variate normal distribution as they relate to factor analysis:
  * [Elliptic Geometry of Multivariate Normal Distribution (Gaussian Distribution)](https://nbviewer.jupyter.org/github/cs224/dev-meetup-dimensionality-reduction/blob/master/factor-analysis-multi-variate-normal-elliptic-geometry.ipynb)
