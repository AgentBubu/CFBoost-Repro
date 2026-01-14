import os
from time import time
import copy
import pickle
import argparse
from time import strftime
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
import pandas as pd
from scipy.sparse import csr_matrix, rand as sprand
from scipy.special import softmax
from scipy.special import log_softmax
from tqdm import tqdm
from past.builtins import range

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from utils import Tool
from utils import MP_Utility

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning, module="resource_tracker")

class MF_CFAdaBoost(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MF_CFAdaBoost, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_df = dataset.train_df

        self.iters = model_conf['iters']
        self.neg_sample_num = model_conf['neg_sample_num']
        self.neg_sample_rate_eval = model_conf['neg_sample_rate_eval']
        self.beta1 = model_conf['beta1']
        self.beta2 = model_conf['beta2']
        self.tau = model_conf['tau']
        self.model_conf = model_conf
        self.device = device

        # LOAD DATA
        # Using the modified filename as per your request
        help_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name)
        self.test_like_item = np.load(help_dir + '/user_test_like.npy', allow_pickle=True)

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']
        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir

        self.time = strftime('%Y%m%d-%H%M')
        similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
        if not os.path.exists(similarity_dir):
            os.mkdir(similarity_dir)

        s_file = os.path.join(similarity_dir, 'MF_CFadaboost_records')
        if not os.path.exists(s_file):
            os.mkdir(s_file)
        similarity_file = os.path.join(s_file, self.time + '_MF_CFrecord_scores')
        if not os.path.exists(similarity_file):
            os.mkdir(similarity_file)
        best_result = None

        start = time()
        m = self.num_users
        n = self.num_items
        
        self.stumps = []
        # === DESIGN 1: Initialize list to store scalar weights ===
        self.stump_weights = [] 
        
        self.sample_weights = np.ones(shape=(m, n)) / (m * n)

        for t in tqdm(range(self.iters)):
            # fitting base learner
            es = copy.deepcopy(early_stop)

            curr_sample_weights = self.sample_weights
            stump = MF(dataset, self.model_conf, self.device)
            stump.train_model(dataset, evaluator, es, logger, config, similarity_file,
                              curr_sample_weights * (self.num_users * self.num_items))

            user_err_list, item_err_list = [], []
            for i in range(self.neg_sample_num):
                user_list, item_list, label_list = MP_Utility.negative_sampling(self.num_users,
                                                                                self.num_items,
                                                                                self.train_df[0],
                                                                                self.train_df[1],
                                                                                self.neg_sample_rate_eval)

                u_emd, v_emd = stump.user_factors.weight.detach().cpu().numpy(), stump.item_factors.weight.detach().cpu().numpy()
                u, v = u_emd[user_list], v_emd[item_list]

                # binary cross entropy
                rec_list = (u * v).sum(axis=1)
                sig_rec = 1 / (1 + np.exp(-rec_list)) # sigmoid
                total_err = -(label_list * np.log(sig_rec) + (1 - label_list) * np.log(1 - sig_rec))

                user_err = np.zeros(self.num_users)
                item_err = np.zeros(self.num_items)
                np.add.at(user_err, user_list, total_err)
                np.add.at(item_err, item_list, total_err)

                user_err /= np.bincount(user_list)
                item_err /= np.bincount(item_list)
                user_err_list.append(user_err)
                item_err_list.append(item_err)

            user_err_mean = np.mean(user_err_list, axis=0).reshape(-1, 1)
            item_err_mean = np.mean(item_err_list, axis=0).reshape(-1, 1)

            # Calculate Error Matrix
            err = np.matmul(user_err_mean, item_err_mean.T) 
            np.clip(err, 1e-15, 1 - 1e-15)

            # ==========================================================
            # === DESIGN 1 LOGIC (CFAdaBoost) ==========================
            # ==========================================================
            
            # 1. Calculate Total Weighted Error (Epsilon)
            # This collapses the matrix into a single number representing global error
            epsilon = (curr_sample_weights * err).sum()

            # 2. Calculate SCALAR Alpha (stump_weight)
            # Using the formula found in the original comments: log(1 - epsilon) + constant
            stump_weight = np.log(1 - epsilon) + 1.6 
            
            # 3. Store the scalar weight for prediction later
            self.stump_weights.append(stump_weight)

            # 4. Update Sample Weights using the scalar alpha
            new_sample_weights = curr_sample_weights * np.exp(stump_weight * err)

            # ==========================================================

            # Normalize sample weights to sum to 1
            new_sample_weights = (new_sample_weights / new_sample_weights.sum())
            self.sample_weights = new_sample_weights

            self.stumps.append(stump)


        test_score_output, ndcg_test_all = evaluator.evaluate_full_boost(self)
        mf_boost_file = os.path.join(similarity_dir, 'MF_CFadaboost_scores')
        if not os.path.exists(mf_boost_file):
            os.mkdir(mf_boost_file)
        with open(os.path.join(mf_boost_file, self.time + '_boost_scores.npy'), 'wb') as f:
            np.save(f, ndcg_test_all)

        # === ADDED FOR TABLES 3 & 4 ===
        # Calculate and Save Item Scores (MDG)
        item_scores = self.calculate_item_scores(k=20)
        
        # Calculate average for Main.py display
        avg_mdg = np.mean(item_scores)
        test_score_output['MDG@20'] = avg_mdg 

        with open(os.path.join(mf_boost_file, self.time + '_ITEM_scores.npy'), 'wb') as f:
            np.save(f, item_scores)
        # ==============================

        total_train_time = time() - start

        return test_score_output, total_train_time

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            Rec = self.predict_helper()
            eval_output = Rec[user_ids, :]
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)] = float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output

    def predict_helper(self):
        rec = None
        
        # ==========================================================
        # === DESIGN 1 PREDICTION (Global Weights) =================
        # ==========================================================
        
        # Convert the list of scalar weights to a tensor
        stump_weights_tensor = torch.tensor(self.stump_weights, dtype=torch.float32).to(self.device)
        
        # Normalize weights using Softmax (as per Design 1 logic)
        stump_weights_tensor = F.softmax(stump_weights_tensor / self.tau, dim=0)

        for i, stump in enumerate(self.stumps):
            # We use the global scalar weight for the whole matrix
            if i == 0:
                rec = stump_weights_tensor[i] * stump.get_rec_tensor()
            else:
                rec += stump_weights_tensor[i] * stump.get_rec_tensor()

        return rec.detach().cpu().numpy()

    # === NEW FUNCTION ADDED FOR TABLES 3 & 4 ===
    def calculate_item_scores(self, k=20):
        """
        Calculates MDG@K for ITEMS using Design 1 (CFAdaBoost) logic.
        """
        self.eval()
        print("Calculating Item MDG scores for CFAdaBoost... (This takes a moment)")
        
        num_users = self.num_users
        num_items = self.num_items
        batch_size = 1024
        
        item_discounted_gains = np.zeros(num_items)
        item_total_likes = np.zeros(num_items)
        
        test_like = self.test_like_item 
        
        # Calculate denominator
        for u_id in range(num_users):
            items_liked = test_like[u_id]
            if len(items_liked) > 0:
                item_total_likes[items_liked] += 1

        with torch.no_grad():
            # Prepare Global Weights (Scalar per model)
            stump_weights_tensor = torch.tensor(self.stump_weights, dtype=torch.float32).to(self.device)
            stump_weights_tensor = F.softmax(stump_weights_tensor / self.tau, dim=0)

            users = np.arange(num_users)
            for start in range(0, num_users, batch_size):
                end = min(start + batch_size, num_users)
                batch_users = users[start:end]
                
                # === CFAdaBoost Design 1 Prediction Logic for Batch ===
                batch_rec = None
                for i, stump in enumerate(self.stumps):
                    # Global Scalar Weight
                    alpha = stump_weights_tensor[i]
                    
                    # Base learner prediction
                    u_emb = stump.user_factors(torch.tensor(batch_users).to(self.device))
                    i_emb = stump.item_factors.weight
                    score = torch.matmul(u_emb, i_emb.T)
                    
                    if batch_rec is None:
                        batch_rec = alpha * score
                    else:
                        batch_rec += alpha * score
                
                batch_scores = batch_rec.cpu().numpy()
                # ======================================================
                
                # Top-K Calculation
                ind = np.argpartition(batch_scores, -k)[:, -k:]
                topk_scores = np.array([batch_scores[row, ind[row]] for row in range(len(batch_scores))])
                sorted_idx_idx = np.argsort(-topk_scores, axis=1)
                topk_indices = np.array([ind[row, sorted_idx_idx[row]] for row in range(len(batch_scores))])
                
                # MDG Calculation
                for idx, u_id in enumerate(batch_users):
                    true_items = set(test_like[u_id])
                    if not true_items: continue
                    
                    ranking = topk_indices[idx]
                    for rank, item_id in enumerate(ranking):
                        if item_id in true_items:
                            gain = 1.0 / np.log2(rank + 2)
                            item_discounted_gains[item_id] += gain

        item_mdg = np.zeros(num_items)
        with np.errstate(divide='ignore', invalid='ignore'):
            item_mdg = item_discounted_gains / item_total_likes
            item_mdg[item_total_likes == 0] = 0.0
            
        return item_mdg


class MF(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MF, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.display_step = model_conf['display_step']
        self.hidden_neuron = model_conf['emb_dim']
        self.neg_sample_rate = model_conf['neg_sample_rate']

        self.batch_size = model_conf['batch_size']
        self.regularization = model_conf['reg']
        self.lr = model_conf['lr']
        self.train_df = dataset.train_df
        self.device = device
        self.loss_function = torch.nn.MSELoss()
        
        # print('******************** MF ********************')
        self.user_factors = torch.nn.Embedding(self.num_users, self.hidden_neuron) 
        self.item_factors = torch.nn.Embedding(self.num_items, self.hidden_neuron) 
        nn.init.xavier_normal_(self.user_factors.weight)
        nn.init.xavier_normal_(self.item_factors.weight)
        
        self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() + LA.norm(self.item_factors.weight.data, 'fro').item())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.regularization)
        self.time = strftime('%Y%m%d-%H%M')

        # print('********************* MF Initialization Done *********************')
        self.to(self.device)

    def forward(self, user, item):
        # Get the dot product per row
        u = self.user_factors(user)
        v = self.item_factors(item)
        x = (u * v).sum(1)
        return x

    def train_model(self, dataset, evaluator, early_stop, logger, config, similarity_file, sample_weights):
        exp_config = config['Experiment']
        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir

        best_result = None

        start = time()
        for epoch_itr in range(1, num_epochs + 1):
            self.train()
            ndcg_test_all = None
            epoch_cost = 0.

            self.user_list, self.item_list, self.label_list, self.weights = MP_Utility.negative_sampling_boost(
                self.num_users,
                self.num_items,
                self.train_df[0],
                self.train_df[1],
                self.neg_sample_rate,
                sample_weights)
            
            batch_loader = DataBatcher(np.arange(len(self.user_list)), batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                tmp_cost = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
                                            self.label_list[batch_idx], self.weights[batch_idx])
                epoch_cost += tmp_cost
                # if verbose and (b + 1) % verbose == 0:
                #     print('batch %d / %d loss = %.4f' % (b + 1, num_batches, tmp_cost))
            epoch_train_time = time() - epoch_train_start
            epoch_info = ['epoch=%3d' % epoch_itr, 'loss=%.3f' % epoch_cost, 'train time=%.2f' % epoch_train_time]

            ## evaluation
            if (epoch_itr >= test_from and epoch_itr % test_step == 0) or epoch_itr == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                test_score = evaluator.evaluate_vali(self)

                updated, should_stop = early_stop.step(test_score, epoch_itr)
                test_score_output = evaluator.evaluate(self)
                test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output if k.startswith('NDCG')]

                if should_stop:
                    # logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        best_result = test_score_output

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch_itr % print_step == 0:
                logger.info(', '.join(epoch_info))

        return best_result, time() - start

    def train_batch(self, user_input, item_input, label_input, weights):
        # reset gradients
        self.optimizer.zero_grad()
        users = torch.Tensor(user_input).int().to(self.device)
        items = torch.Tensor(item_input).int().to(self.device)
        labels = torch.Tensor(label_input).float().to(self.device)
        weights = torch.Tensor(weights).float().to(self.device)
        total_loss = 0

        self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() + LA.norm(self.item_factors.weight.data, 'fro').item())
        y_hat = self.forward(users, items)

        # binary cross entropy loss
        y_hat_sig = F.sigmoid(y_hat)
        loss = -(weights * (labels * torch.log(y_hat_sig) + (1 - labels) * torch.log(1 - y_hat_sig))).sum() 

        added_loss = loss + self.regularization_term

        total_loss += added_loss
        loss.backward()
        self.optimizer.step()

        return total_loss

    def get_rec_tensor(self):
        P, Q = self.user_factors.weight, self.item_factors.weight
        Rec = torch.matmul(P, Q.T)
        return Rec
    
    def get_rec(self):
        P, Q = self.user_factors.weight, self.item_factors.weight
        P = P.detach().cpu().numpy()
        Q = Q.detach().cpu().numpy()
        Rec = np.matmul(P, Q.T)
        return Rec

    def make_records(self):  
        P, Q = self.user_factors.weight, self.item_factors.weight
        P = P.detach().cpu().numpy()
        Q = Q.detach().cpu().numpy()
        similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
        if not os.path.exists(similarity_dir):
            os.mkdir(similarity_dir)
        similarity_file = os.path.join(similarity_dir, 'PC_CFA_saves')
        if not os.path.exists(similarity_file):
            os.mkdir(similarity_file)
        with open(os.path.join(similarity_file,'P_CFA.npy'), 'wb') as f:
            np.save(f, P)
        with open(os.path.join(similarity_file,'Q_CFA.npy'), 'wb') as f:
            np.save(f, Q)

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            Rec = self.get_rec()
            eval_output = Rec[user_ids, :]
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)] = float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output

# ORIGINAL 
# import os
# from time import time
# import copy
# import pickle
# import argparse
# from time import strftime
# import warnings

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import linalg as LA
# import pandas as pd
# from scipy.sparse import csr_matrix, rand as sprand
# from scipy.special import softmax
# from scipy.special import log_softmax
# from tqdm import tqdm
# from past.builtins import range

# from base.BaseRecommender import BaseRecommender
# from dataloader.DataBatcher import DataBatcher
# from utils import Tool
# from utils import MP_Utility

# # Filter warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="resource_tracker")

# # Changes made to switch to Design 1:
# # Training Logic: Instead of calculating a weight matrix (Equation 2 in the paper), it now calculates a scalar weight (Total Error) and appends it to self.stump_weights.
# # Prediction Logic: Instead of loading .npy files to reconstruct weights per user, it uses the saved scalar stump_weights list with Softmax normalization (as per the commented-out instructions in the original file).
# # Efficiency: I removed the file saving/loading of user/item vectors (u_*.npy) since Design 1 does not strictly need them for prediction, making this version faster.

# class MF_CFAdaBoost(BaseRecommender):
#     def __init__(self, dataset, model_conf, device):
#         super(MF_CFAdaBoost, self).__init__(dataset, model_conf)
#         self.dataset = dataset
#         self.num_users = dataset.num_users
#         self.num_items = dataset.num_items
#         self.train_df = dataset.train_df

#         self.iters = model_conf['iters']
#         self.neg_sample_num = model_conf['neg_sample_num']
#         self.neg_sample_rate_eval = model_conf['neg_sample_rate_eval']
#         self.beta1 = model_conf['beta1']
#         self.beta2 = model_conf['beta2']
#         self.tau = model_conf['tau']
#         self.model_conf = model_conf
#         self.device = device

#         # LOAD DATA
#         # Using the modified filename as per your request
#         help_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name)
#         self.test_like_item = np.load(help_dir + '/user_test_like.npy', allow_pickle=True)

#     def train_model(self, dataset, evaluator, early_stop, logger, config):
#         exp_config = config['Experiment']
#         num_epochs = exp_config['num_epochs']
#         print_step = exp_config['print_step']
#         test_step = exp_config['test_step']
#         test_from = exp_config['test_from']
#         verbose = exp_config['verbose']
#         log_dir = logger.log_dir

#         self.time = strftime('%Y%m%d-%H%M')
#         similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
#         if not os.path.exists(similarity_dir):
#             os.mkdir(similarity_dir)

#         s_file = os.path.join(similarity_dir, 'MF_CFadaboost_records')
#         if not os.path.exists(s_file):
#             os.mkdir(s_file)
#         similarity_file = os.path.join(s_file, self.time + '_MF_CFrecord_scores')
#         if not os.path.exists(similarity_file):
#             os.mkdir(similarity_file)
#         best_result = None

#         start = time()
#         m = self.num_users
#         n = self.num_items
        
#         self.stumps = []
#         # === DESIGN 1: Initialize list to store scalar weights ===
#         self.stump_weights = [] 
        
#         self.sample_weights = np.ones(shape=(m, n)) / (m * n)

#         for t in tqdm(range(self.iters)):
#             # fitting base learner
#             es = copy.deepcopy(early_stop)

#             curr_sample_weights = self.sample_weights
#             stump = MF(dataset, self.model_conf, self.device)
#             stump.train_model(dataset, evaluator, es, logger, config, similarity_file,
#                               curr_sample_weights * (self.num_users * self.num_items))

#             user_err_list, item_err_list = [], []
#             for i in range(self.neg_sample_num):
#                 user_list, item_list, label_list = MP_Utility.negative_sampling(self.num_users,
#                                                                                 self.num_items,
#                                                                                 self.train_df[0],
#                                                                                 self.train_df[1],
#                                                                                 self.neg_sample_rate_eval)

#                 u_emd, v_emd = stump.user_factors.weight.detach().cpu().numpy(), stump.item_factors.weight.detach().cpu().numpy()
#                 u, v = u_emd[user_list], v_emd[item_list]

#                 # binary cross entropy
#                 rec_list = (u * v).sum(axis=1)
#                 sig_rec = 1 / (1 + np.exp(-rec_list)) # sigmoid
#                 total_err = -(label_list * np.log(sig_rec) + (1 - label_list) * np.log(1 - sig_rec))

#                 user_err = np.zeros(self.num_users)
#                 item_err = np.zeros(self.num_items)
#                 np.add.at(user_err, user_list, total_err)
#                 np.add.at(item_err, item_list, total_err)

#                 user_err /= np.bincount(user_list)
#                 item_err /= np.bincount(item_list)
#                 user_err_list.append(user_err)
#                 item_err_list.append(item_err)

#             user_err_mean = np.mean(user_err_list, axis=0).reshape(-1, 1)
#             item_err_mean = np.mean(item_err_list, axis=0).reshape(-1, 1)

#             # Calculate Error Matrix
#             err = np.matmul(user_err_mean, item_err_mean.T) 
#             np.clip(err, 1e-15, 1 - 1e-15)

#             # ==========================================================
#             # === DESIGN 1 LOGIC (CFAdaBoost) ==========================
#             # ==========================================================
            
#             # 1. Calculate Total Weighted Error (Epsilon)
#             # This collapses the matrix into a single number representing global error
#             epsilon = (curr_sample_weights * err).sum()

#             # 2. Calculate SCALAR Alpha (stump_weight)
#             # Using the formula found in the original comments: log(1 - epsilon) + constant
#             stump_weight = np.log(1 - epsilon) + 1.6 
            
#             # 3. Store the scalar weight for prediction later
#             self.stump_weights.append(stump_weight)

#             # 4. Update Sample Weights using the scalar alpha
#             new_sample_weights = curr_sample_weights * np.exp(stump_weight * err)

#             # ==========================================================

#             # Normalize sample weights to sum to 1
#             new_sample_weights = (new_sample_weights / new_sample_weights.sum())
#             self.sample_weights = new_sample_weights

#             self.stumps.append(stump)


#         test_score_output, ndcg_test_all = evaluator.evaluate_full_boost(self)
#         mf_boost_file = os.path.join(similarity_dir, 'MF_CFadaboost_scores')
#         if not os.path.exists(mf_boost_file):
#             os.mkdir(mf_boost_file)
#         with open(os.path.join(mf_boost_file, self.time + '_boost_scores.npy'), 'wb') as f:
#             np.save(f, ndcg_test_all)

#         total_train_time = time() - start

#         return test_score_output, total_train_time

#     def predict(self, user_ids, eval_pos_matrix, eval_items=None):
#         self.eval()
#         batch_eval_pos = eval_pos_matrix[user_ids]
#         with torch.no_grad():
#             Rec = self.predict_helper()
#             eval_output = Rec[user_ids, :]
#             if eval_items is not None:
#                 eval_output[np.logical_not(eval_items)] = float('-inf')
#             else:
#                 eval_output[batch_eval_pos.nonzero()] = float('-inf')
#         self.train()
#         return eval_output

#     def predict_helper(self):
#         rec = None
        
#         # ==========================================================
#         # === DESIGN 1 PREDICTION (Global Weights) =================
#         # ==========================================================
        
#         # Convert the list of scalar weights to a tensor
#         stump_weights_tensor = torch.tensor(self.stump_weights, dtype=torch.float32).to(self.device)
        
#         # Normalize weights using Softmax (as per Design 1 logic)
#         stump_weights_tensor = F.softmax(stump_weights_tensor / self.tau, dim=0)

#         for i, stump in enumerate(self.stumps):
#             # We use the global scalar weight for the whole matrix
#             if i == 0:
#                 rec = stump_weights_tensor[i] * stump.get_rec_tensor()
#             else:
#                 rec += stump_weights_tensor[i] * stump.get_rec_tensor()

#         return rec.detach().cpu().numpy()


# class MF(BaseRecommender):
#     def __init__(self, dataset, model_conf, device):
#         super(MF, self).__init__(dataset, model_conf)
#         self.dataset = dataset
#         self.num_users = dataset.num_users
#         self.num_items = dataset.num_items

#         self.display_step = model_conf['display_step']
#         self.hidden_neuron = model_conf['emb_dim']
#         self.neg_sample_rate = model_conf['neg_sample_rate']

#         self.batch_size = model_conf['batch_size']
#         self.regularization = model_conf['reg']
#         self.lr = model_conf['lr']
#         self.train_df = dataset.train_df
#         self.device = device
#         self.loss_function = torch.nn.MSELoss()
        
#         # print('******************** MF ********************')
#         self.user_factors = torch.nn.Embedding(self.num_users, self.hidden_neuron) 
#         self.item_factors = torch.nn.Embedding(self.num_items, self.hidden_neuron) 
#         nn.init.xavier_normal_(self.user_factors.weight)
#         nn.init.xavier_normal_(self.item_factors.weight)
        
#         self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() + LA.norm(self.item_factors.weight.data, 'fro').item())

#         self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.regularization)
#         self.time = strftime('%Y%m%d-%H%M')

#         # print('********************* MF Initialization Done *********************')
#         self.to(self.device)

#     def forward(self, user, item):
#         # Get the dot product per row
#         u = self.user_factors(user)
#         v = self.item_factors(item)
#         x = (u * v).sum(1)
#         return x

#     def train_model(self, dataset, evaluator, early_stop, logger, config, similarity_file, sample_weights):
#         exp_config = config['Experiment']
#         num_epochs = exp_config['num_epochs']
#         print_step = exp_config['print_step']
#         test_step = exp_config['test_step']
#         test_from = exp_config['test_from']
#         verbose = exp_config['verbose']
#         log_dir = logger.log_dir

#         best_result = None

#         start = time()
#         for epoch_itr in range(1, num_epochs + 1):
#             self.train()
#             ndcg_test_all = None
#             epoch_cost = 0.

#             self.user_list, self.item_list, self.label_list, self.weights = MP_Utility.negative_sampling_boost(
#                 self.num_users,
#                 self.num_items,
#                 self.train_df[0],
#                 self.train_df[1],
#                 self.neg_sample_rate,
#                 sample_weights)
            
#             batch_loader = DataBatcher(np.arange(len(self.user_list)), batch_size=self.batch_size, drop_remain=False, shuffle=True)
#             num_batches = len(batch_loader)
#             # ======================== Train
#             epoch_train_start = time()
#             for b, batch_idx in enumerate(batch_loader):
#                 tmp_cost = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
#                                             self.label_list[batch_idx], self.weights[batch_idx])
#                 epoch_cost += tmp_cost
#                 # if verbose and (b + 1) % verbose == 0:
#                 #     print('batch %d / %d loss = %.4f' % (b + 1, num_batches, tmp_cost))
#             epoch_train_time = time() - epoch_train_start
#             epoch_info = ['epoch=%3d' % epoch_itr, 'loss=%.3f' % epoch_cost, 'train time=%.2f' % epoch_train_time]

#             ## evaluation
#             if (epoch_itr >= test_from and epoch_itr % test_step == 0) or epoch_itr == num_epochs:
#                 self.eval()
#                 # evaluate model
#                 epoch_eval_start = time()

#                 test_score = evaluator.evaluate_vali(self)

#                 updated, should_stop = early_stop.step(test_score, epoch_itr)
#                 test_score_output = evaluator.evaluate(self)
#                 test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output if k.startswith('NDCG')]

#                 if should_stop:
#                     # logger.info('Early stop triggered.')
#                     break
#                 else:
#                     # save best parameters
#                     if updated:
#                         best_result = test_score_output

#                 epoch_eval_time = time() - epoch_eval_start
#                 epoch_time = epoch_train_time + epoch_eval_time

#                 epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
#                 epoch_info += test_score_str
#             else:
#                 epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

#             if epoch_itr % print_step == 0:
#                 logger.info(', '.join(epoch_info))

#         return best_result, time() - start

#     def train_batch(self, user_input, item_input, label_input, weights):
#         # reset gradients
#         self.optimizer.zero_grad()
#         users = torch.Tensor(user_input).int().to(self.device)
#         items = torch.Tensor(item_input).int().to(self.device)
#         labels = torch.Tensor(label_input).float().to(self.device)
#         weights = torch.Tensor(weights).float().to(self.device)
#         total_loss = 0

#         self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() + LA.norm(self.item_factors.weight.data, 'fro').item())
#         y_hat = self.forward(users, items)

#         # binary cross entropy loss
#         y_hat_sig = F.sigmoid(y_hat)
#         loss = -(weights * (labels * torch.log(y_hat_sig) + (1 - labels) * torch.log(1 - y_hat_sig))).sum() 

#         added_loss = loss + self.regularization_term

#         total_loss += added_loss
#         loss.backward()
#         self.optimizer.step()

#         return total_loss

#     def get_rec_tensor(self):
#         P, Q = self.user_factors.weight, self.item_factors.weight
#         Rec = torch.matmul(P, Q.T)
#         return Rec
    
#     def get_rec(self):
#         P, Q = self.user_factors.weight, self.item_factors.weight
#         P = P.detach().cpu().numpy()
#         Q = Q.detach().cpu().numpy()
#         Rec = np.matmul(P, Q.T)
#         return Rec

#     def make_records(self):  
#         P, Q = self.user_factors.weight, self.item_factors.weight
#         P = P.detach().cpu().numpy()
#         Q = Q.detach().cpu().numpy()
#         similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
#         if not os.path.exists(similarity_dir):
#             os.mkdir(similarity_dir)
#         similarity_file = os.path.join(similarity_dir, 'PC_saves')
#         if not os.path.exists(similarity_file):
#             os.mkdir(similarity_file)
#         with open(os.path.join(similarity_file,'P_MF.npy'), 'wb') as f:
#             np.save(f, P)
#         with open(os.path.join(similarity_file,'Q_MF.npy'), 'wb') as f:
#             np.save(f, Q)

#     def predict(self, user_ids, eval_pos_matrix, eval_items=None):
#         self.eval()
#         batch_eval_pos = eval_pos_matrix[user_ids]
#         with torch.no_grad():
#             Rec = self.get_rec()
#             eval_output = Rec[user_ids, :]
#             if eval_items is not None:
#                 eval_output[np.logical_not(eval_items)] = float('-inf')
#             else:
#                 eval_output[batch_eval_pos.nonzero()] = float('-inf')
#         self.train()
#         return eval_output