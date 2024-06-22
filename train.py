import os
import args
import configparser
from configf import Config

def run(args, config):
    import tensorflow as tf
    print("Available Devices: ", tf.config.list_physical_devices())
    
    from preprocessing import drugood, mnist, toy, synthetic_chem
    from models import base_learners, meta_learners
    from utils.utils import check_diversity, evaluate_metrics
    
    dataparser = {
        'drugood': drugood.parse_data,
        'toy': toy.parse_data,
        'mnist': mnist.parse_data,
        'synth_chem': synthetic_chem.parse_data
    }
    # test data is considered to be unseen OOD - see test.py
    train_X, train_Y, val_X, val_Y, test_X, test_Y, scaler = dataparser[args.dataset](args, config)
    
    # betas, supps, random_dirs, model, pca_projs_data, pca_projs_dir = base_learners.get_learners(train_X, train_Y, args, config)
    betas, supps, random_dirs, model, pca_projs_data, pca_projs_dir = base_learners.load_learners(train_X, args, config)
    # evaluate_metrics(train_X, train_Y, model, pca_projs_data, pca_projs_dir, random_dirs, betas, args)
    evaluate_metrics(val_X, val_Y, model, pca_projs_data, pca_projs_dir, random_dirs, betas, args)
    # evaluate_metrics(test_X, test_Y, model, pca_projs_data, pca_projs_dir, random_dirs, betas, args)
    # check_diversity(betas, betas, model, pca_projs, train_X, train_Y)
    check_diversity(betas, betas, model, pca_projs_data, pca_projs_dir, val_X, val_Y)
    
    # ml = meta_learners.get_learners(train_X, train_Y, bl, args, config)
    # evaluate_metrics(ml, train_X, train_Y, args)
    # evaluate_metrics(ml, val_X, val_Y, args)
    
    
    # check_diversity(bl, ml, val_X, val_Y, args)
    
    return

if __name__ == "__main__":
    parser = args.create_parser().parse_args()
    # learner_config = configparser.ConfigParser()
    # learner_config.read(parser.config_path) if parser.config_path else ''
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{parser.device}"
    
    run(parser, Config())
    
    
    
    
    