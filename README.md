# Masters research - Prinavan Pillay (PLLPRI017)
### University of Cape Town

A  recommendation  system  for  insurance  predictions was  designed,  implemented  and tested using a neural network and collaborative filtering approach.  The recommendation engine aims to determine suitable insurance products for new or existing customers,based on their profiles or previous purchases.  The collaborative filtering approach used matrix factorization on an existing user base to provide recommendation scores for new products to existing users.  The content based method uses a neural network architecture which incorporated user features to provide a product recommendation for newusers.  Both  methods  were  deployed  using Tensorflow.  The hybrid approach helps solve for cold start problems where users have no interaction history.

This repository contains the code and notebooks required for model development. The following notebooks and modules can be found:

1. GLM_insurance.ipynb - this contains a simple multinomial logistic regression on the insurance dataset
2. Clustering.ipynb - data exploration with unsupervised clustering techniques
3. nn_preproc.ipynb - this is the preprocessing of data required for neural network model development
4. nn_model.ipynb - the neural network recommendation engine
5. nn_htuning.ipynb - hyperparameter tuning for the neural network recommendation engine
6. CF_wals.ipynb - collaborative filtering recommendation engine including preprocessing, model development and model tunung
7. wals_packaged - packaged python module version for CF_wals.ipynb
8. wals_htune - packaged files required for hyperparameter tuning of WALS recommendation engine
