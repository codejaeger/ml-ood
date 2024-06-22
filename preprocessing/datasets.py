# tabular popular uci, moa (LIT)
# https://aaai.org/ojs/index.php/AAAI/article/view/6004 - Ensembles of Locally Independent Prediction Models

# https://ojs.aaai.org/index.php/AAAI/article/view/17163
# DIBS: Diversity Inducing Information Bottleneck in Model Ensembles
# 32x32 - pca or feature extraction, vgg-10
# cifar-10 (), svhn-mnist

# colormnist - pca
# https://github.com/deeplearning-wisc/Spurious_OOD/blob/pub/datasets/color_mnist.py

# more benchmarks ZooD: Exploiting Model Zoo for Out-of-Distribution Generalization
# benchmarks - RF, XGB, ERM, MLDG, GroupDRO, EoA, SWAD
# DIBS, D-BAT (compare prediction diversity on ood data not feature diversity)


# vlcs, pacs image datasets - feature extraction decaf, resnet
# Meta-forests: Domain generalization on random forests with meta-learning
# https://arxiv.org/pdf/2401.04425

# domainbed maybe
# Diverse Weight Averaging for Out-of-Distribution Generalization
# https://proceedings.neurips.cc/paper_files/paper/2022/file/46108d807b50ad4144eb353b5d0e8851-Paper-Conference.pdf
# ZooD: Exploiting Model Zoo for Out-of-Distribution Generalization
# Ensemble of Averages: Improving Model Selection and Boosting Performance in Domain Generalization

# tabular datasets- kaggle binary classifications
# https://arxiv.org/html/2312.04273v3 - Invariant Random Forest: Tree-Based Model Solution for OOD Generalization


## choose small to medium datasets 
# like medical datasets (drugood) (binary), - basic, druogood benchmarks (big table)
# uci tabular datasets (4), mushroom, ionosphere, sonar, spectf binary and electricity - basic, lit, (medium table), mldg (same networks as lit just different training), eoa (use same networks as lit just change training)
# more (3) tabular datasets binary https://arxiv.org/html/2312.04273v3 - irf (special table sota tree based model comparison) (small table)
# images (vlcs decaf, - zood, use domainbed (big table)
# pacs resnet50, - zood, use domainbed
# colormnist-mnist) - use domainbed

# synthetic datasets
# toy (#lobes, vs dimensions, random angles) (big table)

#  feature extractors - vgg 19 resnet 18 (mnist cifar), 

## basic comp rf, xgboost, rbf kernel-svm ensemble, 

# lit - https://github.com/dtak/lit
# mldg - https://github.com/HAHA-DL/MLDG?tab=readme-ov-file
# eoa - https://github.com/salesforce/ensemble-of-averages?tab=readme-ov-file

# Todos:
# 1. Get uci datasets in repo
# 2. read drugood paper