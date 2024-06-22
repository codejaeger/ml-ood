from xgboost import XGBClassifier
from scipy.linalg import null_space
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import tensorflow as tf
import pickle


from utils.utils import pca, project_and_filter, linear_coefs
from utils.rff import define_rand_feats

def get_models(X, Y, pca_projs_data, pca_projs_dir, dirs, model, args):
    betas = []
    supps = []
    i = 0
    for dir in dirs:
        if i % 25 == 0: print(f"Step {i}")
        rand_feats = model(np.array(X@pca_projs_data))
        if args.directional_subspace == "rff":
            X_sub, X_ids = project_and_filter(rand_feats, dir, args.directional_percentile)
            Y_sub = Y[X_ids]
        elif args.directional_subspace == "data":
            _, X_ids = project_and_filter(X, dir, args.directional_percentile)
            Y_sub = Y[X_ids]
            X_sub = rand_feats[X_ids]
        elif args.directional_subspace == "pca":
            _, X_ids = project_and_filter(X@pca_projs_dir, dir, args.directional_percentile)
            Y_sub = Y[X_ids]
            X_sub = rand_feats[X_ids]
        print("Filtered points ", len(X_sub))
        
        # X_sub = np.load('scripts/x_sub.npy')
        # Y_sub = np.load('scripts/y_sub.npy')
        # xid = np.load('scripts/xid.npy')
        # beta, supp = linear_coefs(X_sub, xid, Y_sub, args)
        beta, supp = linear_coefs(X_sub, np.argwhere(X_ids), Y_sub, args)
        betas.append(beta)
        supps.append(supp)
        i += 1
        if i == len(dirs) - 1: print(f"Done")

    betas = np.array(betas)

    return betas, supps

def get_learners(train_X, train_Y, args, config):
    np.random.seed(74)
    pca_projs_data, pca_projs_dir = pca(train_X, args, [args.dim_reduction_data, args.directional_pca_dim])
    if not args.use_pca:
        pca_projs_data = tf.constant(np.identity(train_X.shape[-1]))
    if args.directional_subspace != 'pca':
        pca_projs_dir = tf.constant(np.identity(train_X.shape[-1]))
    X_prjs = np.array(train_X @ pca_projs_data)
    
    model = define_rand_feats(X_prjs, args.rff_features, args.rff_bandwidth, args)
    print("Random feat dims: ", model(X_prjs).shape)
    print("Random feat mean: ", tf.reduce_mean(model(X_prjs), axis=0))
    print("Random feat variance: ", tf.math.reduce_std(model(X_prjs), axis=0))
    if model(X_prjs).shape[-1] > 6000:
        assert False, "Random feature dimension too high!!"

    N = args.num_base_learners
    if args.directional_subspace == "rff":
        d = model(X_prjs).shape[-1]
    elif args.directional_subspace == "data":
        d = train_X.shape[-1]
    elif args.directional_subspace == "pca":
        d = pca_projs_dir.shape[-1]
    random_dirs = np.random.randn(N, d)

    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)

    # the percentile data increases diversity without reducing accuracy since the same max margin svm classifier is learnt for the subset data 
    # as well; the subset data leads to better local generalisation
    betas, supps = get_models(train_X, train_Y, pca_projs_data, pca_projs_dir, random_dirs, model, args)
    betas = tf.squeeze(betas)
    random_dirs = tf.constant(random_dirs)
    print("Betas shape: ", betas.shape)
    print("Random dirs shape: ", random_dirs.shape)
    
    save_learners(betas, supps, random_dirs, model, pca_projs_data, pca_projs_dir, args)
    
    return betas, supps, random_dirs, model, pca_projs_data, pca_projs_dir

def save_learners(betas, supps, random_dirs, model, pca_projs_data, pca_projs_dir, args):
    np.save(f'saves/{args.dataset}-random_dirs-{args.base_learner}-{args.directional_subspace}.npy', random_dirs)
    np.save(f'saves/{args.dataset}-betas-{args.base_learner}-{args.directional_subspace}.npy', betas)
    np.save(f'saves/{args.dataset}-Ws-{args.base_learner}-{args.directional_subspace}.npy', model.Ws)
    np.save(f'saves/{args.dataset}-pca_data-{args.base_learner}-{args.directional_subspace}.npy', pca_projs_data)
    np.save(f'saves/{args.dataset}-pca_dir-{args.base_learner}-{args.directional_subspace}.npy', pca_projs_dir)

    # print(f'saves/{args.dataset}-random_dirs-{args.base_learner}-{args.directional_subspace}.npy')
    # print(f'saves/{args.dataset}-betas-{args.base_learner}-{args.directional_subspace}.npy')
    # print(f'saves/{args.dataset}-Ws-{args.base_learner}-{args.directional_subspace}.npy')
    # print(f'saves/{args.dataset}-pca-{args.base_learner}-{args.directional_subspace}.npy')

    with open(f'saves/{args.dataset}-supps-{args.base_learner}-{args.directional_subspace}.npy', "wb") as fp:
        pickle.dump(supps, fp)

def load_learners(X, args, config):
    random_dirs = tf.constant(np.load(f'saves/{args.dataset}-random_dirs-{args.base_learner}-{args.directional_subspace}.npy'))
    # random_dirs = tf.constant(np.load(f'scripts/chem-random_dirs-svm-best1.npy'))
    betas = tf.constant(np.load(f'saves/{args.dataset}-betas-{args.base_learner}-{args.directional_subspace}.npy'))
    # betas = tf.constant(np.load(f'scripts/chem-betas-svm-best1.npy'))
    model = define_rand_feats(X, 2, args.rff_bandwidth, args) # any number works here since we reset Ws below
    model.Ws = tf.constant(np.load(f'saves/{args.dataset}-Ws-{args.base_learner}-{args.directional_subspace}.npy'))
    # model.Ws = tf.constant(np.load(f'scripts/chem-Ws-svm-best1.npy'))
    pca_projs_data = tf.constant(np.load(f'saves/{args.dataset}-pca_data-{args.base_learner}-{args.directional_subspace}.npy'))
    # pca_projs_data = tf.constant(np.load(f'scripts/chem-pca_projs_data-svm-best1.npy'))
    pca_projs_dir = tf.constant(np.load(f'saves/{args.dataset}-pca_dir-{args.base_learner}-{args.directional_subspace}.npy'))
    # pca_projs_dir = tf.constant(np.load(f'scripts/chem-pca_projs_dir-svm-best1.npy'))
    
    with open(f'saves/{args.dataset}-supps-{args.base_learner}-{args.directional_subspace}.npy', "rb") as fp:
    # with open(f'scripts/chem-supps-svm-best.npy', "rb") as fp:
        supps_b = pickle.load(fp)

    # random_dirs = tf.constant(np.load(f'scripts/toy-random_dirs-svm1.npy'))
    # betas = tf.squeeze(tf.constant(np.load('scripts/toy-betas-svm1.npy')))
    # model = define_rand_feats(X, 2, args.rff_bandwidth, args) # any number works here since we reset Ws below
    # model.Ws = tf.constant(np.load(f'scripts/toy-Ws-svm1.npy').astype(np.float64))
    # pca_projs = np.identity(len(tf.constant(np.load(f'scripts/toy-pca-svm1.npy'))))
    
    # with open(f'scripts/supps.npy', "rb") as fp:
    #     supps_b = pickle.load(fp)
    
    print("Betas shape: ", betas.shape)
    print("Random dirs shape: ", random_dirs.shape)
    var = tf.math.reduce_variance(betas, axis=0)
    mean_var = tf.reduce_mean(var)
    print("Min var: ", np.min(var), "Max var: ", np.max(var), "Mean var: ", mean_var)
    if args.base_learner == 'svc':
        supps_0, supps_1 = supps_b[int(np.random.rand()*len(betas))]
        print("Support point shape: ", supps_0.shape)
    return betas, supps_b, random_dirs, model, pca_projs_data, pca_projs_dir

def get_margins(X, pca_projs, wgts, model):
    rand_feats = model(tf.cast(X@pca_projs, dtype=tf.float32))
    sd = (np.concatenate([np.ones((X.shape[0], 1)), rand_feats], axis=-1) @ wgts.numpy().T) * 1.0
    return sd[:]
