import numpy as np
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from utils.rff import define_rand_feats

def get_preds(feats, betas):
    """
    Args:
    randfeats: N x d
    betas: M x d
    Return:
    preds: N x M - each beta predicts on each instance
    """
    sd = (1 / (1 + np.exp(-np.concatenate([np.ones((feats.shape[0], 1)), feats], axis=-1) @ betas.numpy().T)))
    return sd[:]

def aggregate_preds(preds, weights=None):
    if weights is not None:
        mean_pred = np.sum(weights * preds, axis=-1, keepdims=False)
    else:
        mean_pred = np.mean(preds, axis=-1, keepdims=False)
  
    std_pred = np.std(preds, axis=-1, keepdims=False)
    return np.float32(mean_pred), np.float32(std_pred)

def get_preds_and_aggregate(X, model, pca_projs_data, pca_projs_dir, dirs, betas, args):
    # randfeats = tf.constant(np.load('scripts/extern_rf_X.npy'))
    # X = np.load('scripts/extern_X.npy')
    X_prjs = X @ pca_projs_data
    # model = define_rand_feats(X, 2, args.rff_bandwidth, args)
    # # model = define_rand_feats(X, 2, args.rff_bandwidth, args) # any number works here since we reset Ws below
    # model.Ws = tf.constant(np.load(f'scripts/Ws.npy').astype(np.float64))
    randfeats = model(tf.cast(X_prjs, tf.float32))
    # print(np.mean(randfeats - randfeats_.numpy()))
    # exit()
    
    preds = get_preds(randfeats, betas)
    # projection_func = lambda a, b: np.dot(a, tf.transpose(b))
    projection_func = lambda a, b: np.dot(tf.linalg.normalize(a, axis=-1)[0], tf.transpose(tf.linalg.normalize(b, axis=-1)[0]))
    if args.directional_subspace == "rff":
        projs = projection_func(randfeats, dirs)
    elif args.directional_subspace == "data":
        projs = projection_func(X, dirs)
    elif args.directional_subspace == "pca":
        projs = projection_func(X @ pca_projs_dir, dirs)
    
    thresh = np.percentile(projs, 100 - args.directional_percentile, axis=-1)
    if args.ensembling_strategy == 1:
        wghts = (projs >= thresh[:, None]).astype(np.float64)                # directional ensembling    
    elif args.ensembling_strategy == 2:
        wghts = np.ones_like(projs >= thresh[:, None]).astype(np.float64)    # simple averaging
    elif args.ensembling_strategy == 3:
        wghts = (projs >= thresh[:, None]) * tf.nn.softmax(projs*(projs >= thresh[:, None]), axis=-1)   # weighted directional averaging

    wghts /= np.sum(wghts, axis=-1, keepdims=True)
    return aggregate_preds(preds, wghts)
    
    # randfeats = tf.constant(np.load('scripts/extern_rf_X.npy'))
    # X = tf.constant(np.load('scripts/extern_X.npy'))
    # dirs = tf.constant(np.load('scripts/random_d.npy'))
    # betas = tf.constant(np.load('scripts/betas.npy'))
    # print(randfeats.shape, betas.shape)
    # return get_preds_and_aggregate_sorted1(randfeats, X, dirs, betas)

def std_thresholding(preds, y, sp_rand, args):
    threshs = sp_rand
    std_threshs = np.linspace(np.min(threshs), np.max(threshs), 50) # Diff std. dev. thresholds (20 of them in this case)
    reject_rate = [1 - np.mean((threshs<=s)) for s in std_threshs] # Portion of instances rejected @ each std threshold
    accus = [np.mean((preds==y)[(threshs<=s)]) for s in std_threshs] # Acc @ each std thresh.
    tps = [np.sum(((y)*(preds==y))[(threshs<=s)]) for s in std_threshs]  # correct and positive
    fps = [np.sum(((preds)*(preds!=y))[(threshs<=s)]) for s in std_threshs]  # incorrect and predicted positive
    pos = np.sum(y)
    recall = [tp/pos for tp in tps]
    precision = [tp/(tp+fp) for tp, fp in zip(tps, fps)]
    
    plt.figure()
    plt.plot(recall, precision, marker='+', c='orange')
    plt.xticks(np.arange(0, 1.01, step=0.1))
    plt.xticks(np.arange(0, 1.01, step=0.05), minor=True)
    plt.yticks(np.arange(.2, 1.01, step=0.05))
    plt.grid(True, which='both')
    plt.xlabel('Recall ({} Positive)'.format(int(pos)))
    plt.ylabel('Precision')
    plt.title('Precision vs Recall by Thresholding Ensemble Std')
    plt.savefig(f"outputs/{args.dataset}/std_thresh.png")
    

def pred_acc(X, Y, model, pca_projs_data, pca_projs_dir, dirs, betas, args):

    mean_probs, std_probs = get_preds_and_aggregate(X, model, pca_projs_data, pca_projs_dir, dirs, betas, args)
    # Y = np.load('scripts/Y.npy').astype(np.float32)
    preds = np.float32(mean_probs > 0.5)

    print("First 10 Predictions: ", preds[:10])
    print("Total Positive Preds: ", sum(preds))
    print("Total Preds: ", len(preds))
    print("% Positive Preds: ", sum(preds) / len(preds))
    print()
    print("First 10 Ground Truth: ", Y[:10])
    print("Total Positive Ground Truth: ", sum(Y))
    print("Total Ground Truth: ", len(Y))
    print("% Positive Ground Truth: ", sum(Y) / len(Y))
    print()
    print("Accuracy: ", np.sum(preds == Y) / len(preds))
    
    std_thresholding(preds, Y, std_probs, args)
    auprc(Y, mean_probs, args)
    auroc(Y, mean_probs, args)

def auprc(y, preds, args):
    p, r, thres = precision_recall_curve(y, preds)
    plt.figure()
    plt.plot(r, p)
    plt.savefig(f"outputs/{args.dataset}/pr auc.png")
    # m = tf.keras.metrics.AUC(curve='PR')
    print("PR AUC: ", auc(r, p))

    print("PR AUC (<0.1R): ", auc(r[r<0.1], p[r<0.1])/np.max(r[r<0.1]))
    print("PR AUC (<0.2R): ", auc(r[r<0.2], p[r<0.2])/np.max(r[r<0.2]))

def auroc(y, preds, args):
    print("AUROC: ", roc_auc_score(y, preds))