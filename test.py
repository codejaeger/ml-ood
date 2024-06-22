## Note: USE THIS SCRIPT AFTER PERFORMING HYPERPARAMETER TUNING ON ALL DATASETS.
## USE ONLY WITH SAVED MODELS/WEIGHTS AND FOR EVALUATION ONLY.

import os
import configparser

from args import create_parser

def run(args, config):
    import tensorflow as tf
    print("GPU Devices available: ", tf.config.list_physical_devices())
    
    from preprocessing import drugood, mnist, toy, synthetic_chem
    from models import base_learners, meta_learners
    from utils.utils import check_diversity, evaluate_metrics
    
    dataparser = {
        'drugood': drugood.parse_data,
        'toy': toy.parse_data,
        'mnist': mnist.parse_data,
        'synth-chem': synthetic_chem.parse_data
    }
    # test data is considered to be unseen OOD - see test.py
    _, _, _, _, test_X, test_Y = dataparser[args.dataset](args)
    
    bl = base_learners.load_learners(args, config)
    evaluate_metrics(bl, test_X, test_Y, args)
    
    ml = meta_learners.load_learners(args, config)
    evaluate_metrics(ml, test_X, test_Y, args)
    
    check_diversity(bl, ml, test_X, test_Y, args)
    return


if __name__ == "__main__":
    parser = create_parser().parse_args()
    learner_config = configparser.ConfigParser()
    learner_config.read(parser.config_path) if parser.config_path else ''
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{parser.device}"
    
    run(parser, learner_config)
    
    
    
    
    