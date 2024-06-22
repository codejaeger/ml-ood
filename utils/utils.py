import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.metrics import pred_acc
from sklearn.metrics import jaccard_score

def check_diversity(betas1, betas2, model, pca_projs_data, pca_projs_dir, X, Y):
    print("Betas similarity: ", check_simi(betas1, betas2, all_pairs=False))
    print("Betas similarity (all pairs): ", check_simi(betas1, betas2, all_pairs=True))
    # X = tf.constant(np.load('scripts/extern_rf_X.npy'))
    # Y = tf.constant(np.load('scripts/Y.npy'))
    
    randfeats = model(tf.cast(X@pca_projs_data, tf.float32)) # model weights in tf.float32
    y_preds1 = predict(randfeats, betas1)
    y_preds2 = predict(randfeats, betas2)
    check_pred_simi(y_preds1, y_preds2, Y, all_pairs=False)
    check_pred_simi(y_preds1, y_preds2, Y, all_pairs=True)
    
def check_pred_simi(y_pred1, y_pred2, y_true, all_pairs=False):
    if all_pairs:
        y_pred1 = y_pred1[None, :, :]
        y_true1 = y_true[None, :, None]
        y_pred2 = y_pred2[:, None, :]
        y_true2 = y_true[:, None, None]
    else:
        y_true1 = y_true2 = y_true[:, None]
    
    tp1 = np.float32(y_pred1==1) * np.float32(y_true1==1)
    fp1 = np.float32(y_pred1==1) * np.float32(y_true1==0)
    tn1 = np.float32(y_pred1==0) * np.float32(y_true1==0)
    fn1 = np.float32(y_pred1==0) * np.float32(y_true1==1)

    tp2 = np.float32(y_pred2==1) * np.float32(y_true2==1)
    fp2 = np.float32(y_pred2==1) * np.float32(y_true2==0)
    tn2 = np.float32(y_pred2==0) * np.float32(y_true2==0)
    fn2 = np.float32(y_pred2==0) * np.float32(y_true2==1)
    
    mou = tf.keras.metrics.MeanIoU(int(np.max(y_true)+1), axis=-1)
    
    print()
    if all_pairs:
        print("\nAll pairs pred similarity: ")
    print("Pred1 tp {0}, fp {1}, tn {2}, fn {3}".format(np.mean(tp1), np.mean(fp1), np.mean(tn1), np.mean(fn1)))
    print("Pred1 tp {0}, fp {1}, tn {2}, fn {3}".format(np.mean(tp2), np.mean(fp2), np.mean(tn2), np.mean(fn2)))
    print("Matching rate tp {0}, fp {1}, tn {2}, fn {3}".format(np.mean(tp1==tp2), np.mean(fp1==fp2), np.mean(tn1==tn2), np.mean(fn1==fn2)))
    # matching rate high above 50% means all classifiers agree on that class for both correct and incorrect predictions
    # ~ 50% matching rate is random noise
    # print("Jaccard score tp {0}, fp {1}, tn {2}, fn {3}".format(mou(tp1, tp2), mou(fp1, fp2), mou(tn1, tn2), mou(fn1, fn2)))

def agreement(y_pred1, y_pred2, y_true):
    tp1 = np.float32(y_pred1==1) * np.float32(y_true==1)
    fp1 = np.float32(y_pred1==1) * np.float32(y_true==0)
    tn1 = np.float32(y_pred1==0) * np.float32(y_true==0)
    fn1 = np.float32(y_pred1==0) * np.float32(y_true==1)

    tp2 = np.float32(y_pred2==1) * np.float32(y_true==1)
    fp2 = np.float32(y_pred2==1) * np.float32(y_true==0)
    tn2 = np.float32(y_pred2==0) * np.float32(y_true==0)
    fn2 = np.float32(y_pred2==0) * np.float32(y_true==1)
    print("Pred1 tp {0}, fp {1}, tn {2}, fn {3}".format(np.sum(tp1)/len(tp1), np.sum(fp1)/len(tp1), np.sum(tn1)/len(tp1), np.sum(fn1)/len(tp1)))
    print("Pred2 tp {0}, fp {1}, tn {2}, fn {3}".format(np.sum(tp2)/len(tp1), np.sum(fp2)/len(tp1), np.sum(tn2)/len(tp1), np.sum(fn2)/len(tp1)))
    print("Matching rate tp {0}, fp {1}, tn {2}, fn {3}".format(np.sum(tp1==tp2)/len(tp1), np.sum(fp1==fp2)/len(tp1), np.sum(tn1==tn2)/len(tp1), np.sum(fn1==fn2)/len(tp1)))
    print("Jaccard score tp {0}, fp {1}, tn {2}, fn {3}".format(jaccard_score(tp1, tp2), jaccard_score(fp1, fp2), jaccard_score(tn1, tn2), jaccard_score(fn1, fn2)))

def check_simi(a, b, all_pairs=False):
    if all_pairs:
        return tf.keras.metrics.CosineSimilarity(axis=-1)(a[None, :, :], b[:, None, :])
    else:
        return tf.keras.metrics.CosineSimilarity()(a, b)

def evaluate_metrics(X, Y, model, pca_projs_data, pca_projs_dir, dirs, betas, args):
    pred_acc(X, Y, model, pca_projs_data, pca_projs_dir, dirs, betas, args)

def project_and_filter(X, dir, percentile=75):
    projs = np.dot(X, dir)
    thresh = np.percentile(projs, 100 - percentile)
    filtered_idxs = projs >= thresh
    return X[filtered_idxs], filtered_idxs

def linear_coefs(X, X_ids, Y, args):
    """
    Args:
    X: N x d matrix of input features
    Y: N x 1 matrix (column vector) of output response

    Returns:
    Beta: d x 1 matrix of linear coefficients
    """
    if args.base_learner == "log":
        clf = LogisticRegression(random_state=0, solver='liblinear').fit(X, Y)
    elif args.base_learner == "svc":
        clf = SVC(random_state=0, tol=1e-5, kernel='linear').fit(X, Y)
    elif args.base_learner == "lsvc":
        clf = LinearSVC(random_state=0, tol=1e-5).fit(X, Y)
    else:
        assert(False)

    def get_supp(support):
        supps_, n_supps_ = support
        supps_0 = supps_[:n_supps_[0]]
        supps_1 = supps_[n_supps_[0]:]
        return X_ids[supps_0], X_ids[supps_1]

    if args.base_learner == "svc":
        support = (clf.support_, clf.n_support_)
        support = get_supp(support)
    else:
        support = None
    
    print("Classifier score: ", clf.score(X, Y))
    wgts = np.hstack((clf.intercept_[:,None], clf.coef_))
    prd = (1 / (1 + np.exp(-np.concatenate([np.ones((X.shape[0], 1)), X], axis=-1) @ wgts.T)) > 0.5) *1.0
    # print("Prediction acc: ", np.mean(prd[:, 0]==Y)) # must be close to clf score
    return wgts, support

def pca(X, args, dimr=[1.0]):
    with tf.device(f'{args.use_pca_device}:{args.device if args.use_pca_device=="gpu" else 0}'):
        s, u, v = tf.linalg.svd(tf.constant(X))
    # print("Pca check", np.sum(v))
    plt.figure()
    plt.plot(np.arange(len(s)), s)
    plt.savefig(f"outputs/{args.dataset}/pca.png")
    pcas = []
    for _dr in dimr:
        pcas.append(v[:, :int(X.shape[-1]*_dr)])
        print("PCA dimensions: ", pcas[0].shape)
    return pcas

def predict(X, wgts):
    print(X.shape, wgts.shape)
    sd = (1 / (1 + np.exp(-np.concatenate([np.ones((X.shape[0], 1)), X], axis=-1) @ wgts.numpy().T)) > 0.5) *1.0
    return sd[:]

def predict_proba(X, wgts):
    sd = (1 / (1 + np.exp(-np.concatenate([np.ones((X.shape[0], 1)), X], axis=-1) @ wgts.numpy().T)) ) *1.0
    return sd[:]
    

## cross validation


## hyper-parameters search


## 



## 