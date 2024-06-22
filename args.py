import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='toy', type=str, help='dataset to load and evaluate')
    parser.add_argument('--save_metrics', default=True, type=bool, help='save metrics')
    parser.add_argument('--ensembling_strategy', default=1, type=int, help='whether use diretional ensembling (1) or simple averaging (2) or (3) weighted averaging')
    parser.add_argument('--meta_learner', default='vae', type=str, help='meta learning algorithm')
    parser.add_argument('--base_learner', default='svc', type=str, help='base learning algorithm')
    parser.add_argument('--num_base_learners', default=2**8, type=int, help='num base learners')
    parser.add_argument('--use_pca', action='store_true', default=False, help='use dimensional reduction')
    parser.add_argument('--use_pca_device', default='gpu', help='pca on gpu and cpu give different results, on chem data gpu:0 pca gives better results (fix?)')
    parser.add_argument('--dim_reduction_data', default=1.0, type=float, help='dimensionality reduction on data using pca')
    parser.add_argument('--rff_algorithm', default='vanilla', type=str, help='use random fourier feature transform, vanilla (Rahimi and Rect) or reduced (RFSVM)')
    parser.add_argument('--rff_features', default=6, type=float, help='#(rff features)/#(data features)')
    parser.add_argument('--rff_bandwidth', default=[1/4, 1/2, 1, 2, 4], type=list, help='#(rff features)/#(data features)')
    parser.add_argument('--directional_subspace', default='data', help='directional ensembling subspace in rff, pca or original feature space')
    parser.add_argument('--directional_pca_dim', default=1.0, type=float, help='when directional subspace is using pca how many dimensions to use')
    parser.add_argument('--directional_percentile', default=25, type=float, help='percentile projected data used for training base learner')
    parser.add_argument('--config_path', default='', type=str, help='learner config file')
    
    
    parser.add_argument('--device', default=0, type=int, help='gpu device to use for training meta learner')
    parser.add_argument('--outdir', default='outputs', type=str, help='output directory')
    
    return parser
