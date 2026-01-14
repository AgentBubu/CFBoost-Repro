# Import packages
import os
import sys
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as multiprocessing

import utils.Constant as CONSTANT
from dataloader import UIRTDatset
from evaluation import Evaluator
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import warnings
import gc

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

# Set CUDA Device Order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    from experiment import EarlyStop, train_model
    from utils import Config, Logger, ResultTable, make_log_dir

    # 1. Read Configuration
    config = Config(main_conf_path='./', model_conf_path='model_config')

    # 2. Apply Command Line Arguments
    argv = sys.argv[1:]
    if len(argv) > 0:
        cmd_arg = OrderedDict()
        argvs = ' '.join(sys.argv[1:]).split(' ')
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip('-')
            cmd_arg[arg_name] = arg_value
        config.update_params(cmd_arg)

    # 3. Setup GPU
    gpu = config.get_param('Experiment', 'gpu')
    gpu = str(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = config.get_param('Experiment', 'model_name')

    # 4. Logger Setup
    log_dir = make_log_dir(os.path.join('saves', model_name))
    logger = Logger(log_dir)
    config.save(log_dir)

    # 5. Load Dataset
    dataset_name = config.get_param('Dataset', 'dataset')
    dataset = UIRTDatset(**config['Dataset'])

    # 6. Setup Evaluator
    num_users, num_items = dataset.num_users, dataset.num_items
    test_eval_pos, test_eval_target, vali_eval_target, eval_neg_candidates = dataset.test_data()
    
    # We pass None for item_id to save memory unless specifically needed
    test_evaluator = Evaluator(test_eval_pos, test_eval_target, vali_eval_target, 
                               eval_neg_candidates, **config['Evaluator'], 
                               num_users=num_users, num_items=num_items, item_id=None)

    # 7. Early Stop Setup
    early_stop = EarlyStop(**config['EarlyStop'])

    # Save configs to log
    logger.info(config)
    logger.info(dataset)

    # 8. Build Model Dynamically
    import model
    MODEL_CLASS = getattr(model, model_name)
    model = MODEL_CLASS(dataset, config['Model'], device)

    # 9. Train Model
    # Note: test_score returned here contains the Best Result found during training
    test_score, train_time = train_model(model, dataset, test_evaluator, early_stop, logger, config)

    # 10. Log Training Time
    m, s = divmod(train_time, 60)
    h, m = divmod(m, 60)
    logger.info('\nTotal training time - %d:%d:%d(=%.1f sec)' % (h, m, s, train_time))

    # ==============================================================================
    # OUTPUT SECTION: SHOW RESULTS TABLE
    # ==============================================================================
    
    # A. Standard Log Table (Auto-formatted)
    evaluation_table = ResultTable(table_name='Best Result', header=list(test_score.keys()))
    evaluation_table.add_row('Score', test_score)
    logger.info(evaluation_table.to_string())

    # B. Custom Expanded Table (Includes MDG@20 and DG for all K)
    print('\n' + '=' * 140)
    print(f'|| FINAL SUMMARY: {model_name} on {dataset_name}')
    print('=' * 140)
    
    # Define the exact columns you want to see
    keys_to_print = [
        'NDCG@1', 'NDCG@5', 'NDCG@10', 'NDCG@20', 
        'DG@1',   'DG@5',   'DG@10',   'DG@20', 
        'MDG@20'
    ]
    
    # Create Header
    header_str = "|| " + " || ".join([f"{k:<8}" for k in keys_to_print]) + " ||"
    print(header_str)
    print('-' * 140)
    
    # Create Value Row
    val_str = "|| "
    for k in keys_to_print:
        # Get value from test_score dictionary, default to 0.0 if not found
        val = test_score.get(k, 0.0)
        val_str += f"{val:.4f}   || "
    print(val_str)
    print('=' * 140 + '\n')

    logger.info("Saved to %s" % (log_dir))

    # ==============================================================================
    # EXTRACTION SECTION (Legacy support for MultVAE/EASE/LOCA)
    # ==============================================================================
    
    # Extract global model output (for Distillation or Analysis)
    if 'LOCA' not in model_name and (model_name == 'MultVAE' or model_name == 'EASE'):
        output = model.get_output(dataset)

        output_dir = os.path.join(dataset.data_dir, dataset.data_name, 'output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = os.path.join(output_dir, model_name + '_output.p')
        with open(output_file, 'wb') as f:
            pickle.dump(output, f, protocol=4)
        config.save(output_dir)
        print(f"{model_name} output extracted!")

    # Extract Embedding (Specific to MultVAE analysis)
    if model_name == 'MultVAE':
        user_embedding = model.user_embedding(test_eval_pos)

        emb_dir = os.path.join(dataset.data_dir, dataset.data_name, 'embedding')
        if not os.path.exists(emb_dir):
            os.mkdir(emb_dir)
        emb_file = os.path.join(emb_dir, model_name + '_user.p')
        with open(emb_file, 'wb') as f:
            pickle.dump(user_embedding, f, protocol=4)
        config.save(emb_dir)
        print(f"{model_name} embedding extracted!")