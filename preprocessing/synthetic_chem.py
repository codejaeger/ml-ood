import numpy as np
import sklearn
from sklearn import datasets, manifold
from utils.utils import project_and_filter
from utils.plots import plot_2d

def get_data():
    with open('./datasets-ood/chem/train.csv', 'r') as f:
        dataX = np.float32(np.array([line.strip().split(',')[2:] for line in f])[1:])
    with open('./datasets-ood/chem/train.csv', 'r') as f:
        dataY = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])
  
    with open('./datasets-ood/chem/test_ood.csv', 'r') as f:
        test_X = np.float32(np.array([line.strip().split(',')[4:] for line in f])[1:])

    with open('./datasets-ood/chem/test_ood.csv', 'r') as f:
        test_Y = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])
  
    with open('./datasets-ood/chem/val_ood.csv', 'r') as f:
        external_X = np.float32(np.array([line.strip().split(',')[4:] for line in f])[1:])

    with open('./datasets-ood/chem/val_ood.csv', 'r') as f:
        external_Y = np.float32(np.array([line.strip().split(',')[1] for line in f])[1:])
        
    return dataX, dataY, test_X, test_Y, external_X, external_Y

def plot_data(x, y, title, config, args):
    # https://distill.pub/2016/misread-tsne/
    t_sne = manifold.TSNE(
        n_components=2,
        perplexity=10,
        init="random",
        n_iter=500,
        random_state=0,
    )
    y_colors = np.array(['red', 'green'])[y.astype(np.int)]
    S_t_sne = t_sne.fit_transform(x)
    plot_2d(S_t_sne, y_colors, "tsne full data", f"{args.outdir}/synth_chem/fitted_{title}")

def parse_data(args, config):
    import os
    os.makedirs(f'{args.outdir}/synth_chem/', exist_ok=True)
    trX, trY, tX, tY, vX, vY  = get_data()
    scaler = sklearn.preprocessing.StandardScaler(with_std=config.synth_chem.use_std)
    mu_x = np.mean(trX, 0, keepdims=True)
    # sigma_x = np.std(X, 0, keepdims=True)
    sigma_x = np.ones_like(mu_x)
    trX = (trX-mu_x)/sigma_x
    tX = (tX-mu_x)/sigma_x
    vX = (vX-mu_x)/sigma_x
    # scaler.fit(trX)
    # trX = scaler.transform(trX)
    # tX = scaler.transform(tX)
    # vX = scaler.transform(vX)
    print("Data sizes: ", trX.shape, trY.shape, vX.shape, vY.shape, tX.shape, tY.shape)
    # plot_data(trX, trY, "train", config, args)
    # plot_data(vX, vY, "val", config, args)
    # plot_data(tX, tY, "test", config, args)

    # X_sub, X_ids = project_and_filter(trX, np.random.randn(trX.shape[1],), 40)
    # Y_sub = trY[X_ids]
    # plot_data(X_sub, Y_sub, "xproject", config, args)
    return trX, trY, vX, vY, tX, tY, scaler