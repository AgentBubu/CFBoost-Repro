import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import linalg as LA

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from utils import Tool
from utils import MP_Utility

import copy
from past.builtins import range

import pickle
import argparse
import pandas as pd
from scipy.sparse import csr_matrix, rand as sprand
from tqdm import tqdm
from time import strftime

from evaluation.backend import HoldoutEvaluator
# from statistics import mean

# np.random.seed(0)
# torch.manual_seed(0)


class MF(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MF, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.display_step = model_conf['display_step']
        self.hidden_neuron = model_conf['hidden_neuron']
        self.neg_sample_rate = model_conf['neg_sample_rate']

        self.batch_size = model_conf['batch_size']
        self.regularization = model_conf['reg']
        self.lr = model_conf['lr']
        self.train_df = dataset.train_df
        self.device = device
        # self.loss_function = torch.nn.MSELoss()
        # self.loss_function = torch.nn.BCELoss(reduction='sum')
        # self.train_like = dataset.train_like
        # self.test_like = dataset.test_like
        self.user_list, self.item_list, self.label_list = MP_Utility.negative_sampling(self.num_users, self.num_items,
                                                                                    self.train_df[0],
                                                                                    self.train_df[1],
                                                                                    self.neg_sample_rate)
        
        # === ADDED: Load Test Like Item for Item Scoring ===
        help_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name)
        # Using the modified filename format consistent with your setup
        self.test_like_item = np.load(help_dir + '/user_test_like.npy', allow_pickle=True)
        # ===================================================

        print('******************** MF ********************')
        self.user_factors = torch.nn.Embedding(self.num_users, self.hidden_neuron)  # , sparse=True
        # self.user_factors.weight.data.uniform_(-0.05, 0.05)
        self.item_factors = torch.nn.Embedding(self.num_items, self.hidden_neuron)  # , sparse=True
        # self.item_factors.weight.data.uniform_(-0.05, 0.05)
        nn.init.xavier_normal_(self.user_factors.weight)
        nn.init.xavier_normal_(self.item_factors.weight)
        print('P: ', self.user_factors)
        print('Q: ', self.item_factors)
        self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() + LA.norm(self.item_factors.weight.data, 'fro').item())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.regularization)
        self.time = strftime('%Y%m%d-%H%M')

        print('********************* MF Initialization Done *********************')
        self.to(self.device)

    def forward(self, user, item):
        # Get the dot product per row
        u = self.user_factors(user)
        v = self.item_factors(item)
        x = (u * v).sum(1)
        return x

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']
        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir
        
        similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
        if not os.path.exists(similarity_dir):
            os.mkdir(similarity_dir)
        best_result = None

        start = time()
        for epoch_itr in range(1, num_epochs + 1):
            self.train()
            epoch_cost = 0.
            
            self.user_list, self.item_list, self.label_list = MP_Utility.negative_sampling(self.num_users,
                                                                                           self.num_items,
                                                                                           self.train_df[0],
                                                                                           self.train_df[1],
                                                                                           self.neg_sample_rate)
            
            batch_loader = DataBatcher(np.arange(len(self.user_list)), batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                tmp_cost = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
                                                                  self.label_list[batch_idx])
                epoch_cost += tmp_cost
                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, tmp_cost))
            epoch_train_time = time() - epoch_train_start
            epoch_info = ['epoch=%3d' % epoch_itr, 'loss=%.3f' % epoch_cost, 'train time=%.2f' % epoch_train_time]

            ## evaluation
            if (epoch_itr >= test_from and epoch_itr % test_step == 0) or epoch_itr == num_epochs:
                self.eval()
                epoch_eval_start = time()

                test_score = evaluator.evaluate_vali(self)
                updated, should_stop = early_stop.step(test_score, epoch_itr)

                # Get Full Metrics (NDCG, DG, etc)
                test_score_output, ndcg_test_all = evaluator.evaluate_full_boost(self)
                test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output if k.startswith('NDCG')]

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        best_result = test_score_output

                        # 1. Save NDCG scores
                        self.make_records()
                        similarity_file = os.path.join(similarity_dir, 'MF_scores')
                        if not os.path.exists(similarity_file):
                            os.mkdir(similarity_file)
                        with open(os.path.join(similarity_file, self.time + '_mf_scores.npy'), 'wb') as f:
                            np.save(f, ndcg_test_all)
                            
                        # 2. Calculate and Save Item Scores (MDG)
                        item_scores = self.calculate_item_scores(k=20)
                        
                        # --- NEW: Calculate Average MDG for Display ---
                        avg_mdg = np.mean(item_scores)
                        
                         # <<< ADD THIS LINE >>>
                        best_result['MDG@20'] = avg_mdg 
                        # This ensures main.py can see the value!
                        
                        with open(os.path.join(similarity_file, self.time + '_ITEM_scores.npy'), 'wb') as f:
                            np.save(f, item_scores)
                            
                        # --- NEW: FORCE PRINT TABLE WITH MDG ---
                        print("\n" + "="*100)
                        print("||               UPDATED BEST RESULT WITH MDG               ||")
                        print("="*100)
                        print(f"|| NDCG@20 : {test_score_output.get('NDCG@20', 0.0):.4f}  ||  DG@20 : {test_score_output.get('DG@20', 0.0):.4f}  ||  MDG@20 : {avg_mdg:.4f} ||")
                        print("="*100 + "\n")

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch_itr % print_step == 0:
                logger.info(', '.join(epoch_info))

            total_train_time = time() - start

        return best_result, total_train_time
    
    def train_batch(self, user_input, item_input, label_input):
        # reset gradients
        self.optimizer.zero_grad()
        users = torch.Tensor(user_input).int().to(self.device)
        items = torch.Tensor(item_input).int().to(self.device)
        labels = torch.Tensor(label_input).float().to(self.device)
        total_loss = 0

        self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() + LA.norm(self.item_factors.weight.data, 'fro').item())
        y_hat = self.forward(users, items)

        y_hat_sig = F.sigmoid(y_hat)
        loss = -(labels * torch.log(y_hat_sig) + (1 - labels) * torch.log(1 - y_hat_sig)).sum()
        added_loss = loss + self.regularization_term

        total_loss += added_loss
        # backpropagate
        loss.backward()
        # update
        self.optimizer.step()

        return total_loss
    
    # === ADDED: New function to calculate Item MDG scores ===
    def calculate_item_scores(self, k=20):
        """
        Calculates MDG@K for ITEMS for the Standard MF Model.
        """
        self.eval()
        print("Calculating Item MDG scores for MF... (This takes a moment)")
        
        num_users = self.num_users
        num_items = self.num_items
        batch_size = 1024
        
        item_discounted_gains = np.zeros(num_items)
        item_total_likes = np.zeros(num_items)
        
        # Load test data to see which items are actually liked
        test_like = self.test_like_item 
        
        # Calculate how many users like each item (Denominator for MDG)
        for u_id in range(num_users):
            items_liked = test_like[u_id]
            if len(items_liked) > 0:
                item_total_likes[items_liked] += 1

        with torch.no_grad():
            users = np.arange(num_users)
            for start in range(0, num_users, batch_size):
                end = min(start + batch_size, num_users)
                batch_users = users[start:end]
                
                # === MF PREDICTION LOGIC ===
                batch_users_tensor = torch.tensor(batch_users).to(self.device)
                
                # Get Embeddings
                u_emb = self.user_factors(batch_users_tensor)
                i_emb = self.item_factors.weight
                
                # Dot Product
                batch_scores = torch.matmul(u_emb, i_emb.T).cpu().numpy()
                # ===========================
                
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
                            # Log base 2 of (rank + 2) because rank starts at 0 (rank 0 is pos 1)
                            gain = 1.0 / np.log2(rank + 2)
                            item_discounted_gains[item_id] += gain

        item_mdg = np.zeros(num_items)
        with np.errstate(divide='ignore', invalid='ignore'):
            item_mdg = item_discounted_gains / item_total_likes
            item_mdg[item_total_likes == 0] = 0.0
            
        return item_mdg
    # ========================================================

    def get_rec(self):
        P, Q = self.user_factors.weight, self.item_factors.weight
        P = P.detach().cpu().numpy()
        Q = Q.detach().cpu().numpy()
        Rec = np.matmul(P, Q.T)
        return Rec

    def make_records(self):  # record all the results' details into files
        P, Q = self.user_factors.weight, self.item_factors.weight
        P = P.detach().cpu().numpy()
        Q = Q.detach().cpu().numpy()
        similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
        if not os.path.exists(similarity_dir):
            os.mkdir(similarity_dir)
        similarity_file = os.path.join(similarity_dir, 'PC_mf_bce_saves')
        if not os.path.exists(similarity_file):
            os.mkdir(similarity_file)
        with open(os.path.join(similarity_file,'P_MF.npy'), 'wb') as f:
            np.save(f, P)
        with open(os.path.join(similarity_file,'Q_MF.npy'), 'wb') as f:
            np.save(f, Q)
        # return P, Q

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            # eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
            # P = torch.Tensor(self.user_list[user_ids]).int()
            # Q = torch.Tensor(self.item_list[user_ids]).int()
            Rec = self.get_rec()
            eval_output = Rec[user_ids, :]
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)] = float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output


# ORIGINAL FILE: model/MF.py
# import os
# import math
# from time import time

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch import linalg as LA

# from base.BaseRecommender import BaseRecommender
# from dataloader.DataBatcher import DataBatcher
# from utils import Tool
# from utils import MP_Utility

# import copy
# from past.builtins import range

# import pickle
# import argparse
# import pandas as pd
# from scipy.sparse import csr_matrix, rand as sprand
# from tqdm import tqdm
# from time import strftime

# from evaluation.backend import HoldoutEvaluator
# # from statistics import mean

# # np.random.seed(0)
# # torch.manual_seed(0)


# class MF(BaseRecommender):
#     def __init__(self, dataset, model_conf, device):
#         super(MF, self).__init__(dataset, model_conf)
#         self.dataset = dataset
#         self.num_users = dataset.num_users
#         self.num_items = dataset.num_items

#         self.display_step = model_conf['display_step']
#         self.hidden_neuron = model_conf['hidden_neuron']
#         self.neg_sample_rate = model_conf['neg_sample_rate']

#         self.batch_size = model_conf['batch_size']
#         self.regularization = model_conf['reg']
#         self.lr = model_conf['lr']
#         self.train_df = dataset.train_df
#         self.device = device
#         # self.loss_function = torch.nn.MSELoss()
#         # self.loss_function = torch.nn.BCELoss(reduction='sum')
#         # self.train_like = dataset.train_like
#         # self.test_like = dataset.test_like
#         self.user_list, self.item_list, self.label_list = MP_Utility.negative_sampling(self.num_users, self.num_items,
#                                                                                     self.train_df[0],
#                                                                                     self.train_df[1],
#                                                                                     self.neg_sample_rate)
#         print('******************** MF ********************')
#         self.user_factors = torch.nn.Embedding(self.num_users, self.hidden_neuron)  # , sparse=True
#         # self.user_factors.weight.data.uniform_(-0.05, 0.05)
#         self.item_factors = torch.nn.Embedding(self.num_items, self.hidden_neuron)  # , sparse=True
#         # self.item_factors.weight.data.uniform_(-0.05, 0.05)
#         nn.init.xavier_normal_(self.user_factors.weight)
#         nn.init.xavier_normal_(self.item_factors.weight)
#         print('P: ', self.user_factors)
#         print('Q: ', self.item_factors)
#         self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() + LA.norm(self.item_factors.weight.data, 'fro').item())

#         self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.regularization)
#         self.time = strftime('%Y%m%d-%H%M')

#         print('********************* MF Initialization Done *********************')
#         self.to(self.device)

#     def forward(self, user, item):
#         # Get the dot product per row
#         u = self.user_factors(user)
#         v = self.item_factors(item)
#         x = (u * v).sum(1)
#         return x

#     def train_model(self, dataset, evaluator, early_stop, logger, config):
#         exp_config = config['Experiment']
#         num_epochs = exp_config['num_epochs']
#         print_step = exp_config['print_step']
#         test_step = exp_config['test_step']
#         test_from = exp_config['test_from']
#         verbose = exp_config['verbose']
#         log_dir = logger.log_dir
#         # users = np.arange(self.num_users)
#         similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
#         if not os.path.exists(similarity_dir):
#             os.mkdir(similarity_dir)
#         best_result = None

#         start = time()
#         for epoch_itr in range(1, num_epochs + 1):
#             self.train()
#             epoch_cost = 0.
#             # epoch_cost1 = 0.
#             # epoch_cost2 = 0.
#             self.user_list, self.item_list, self.label_list = MP_Utility.negative_sampling(self.num_users,
#                                                                                            self.num_items,
#                                                                                            self.train_df[0],
#                                                                                            self.train_df[1],
#                                                                                            self.neg_sample_rate)
#             # start_time = time() * 1000.0
#             batch_loader = DataBatcher(np.arange(len(self.user_list)), batch_size=self.batch_size, drop_remain=False, shuffle=True)
#             num_batches = len(batch_loader)
#             # ======================== Train
#             epoch_train_start = time()
#             for b, batch_idx in enumerate(batch_loader):
#                 # tmp_cost, tmp_cost1, tmp_cost2 = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
#                 #                                                   self.label_list[batch_idx])
#                 tmp_cost = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
#                                                                   self.label_list[batch_idx])
#                 epoch_cost += tmp_cost
#                 # epoch_cost1 += tmp_cost1
#                 # epoch_cost2 += tmp_cost2
#                 if verbose and (b + 1) % verbose == 0:
#                     print('batch %d / %d loss = %.4f' % (b + 1, num_batches, tmp_cost))
#             epoch_train_time = time() - epoch_train_start
#             epoch_info = ['epoch=%3d' % epoch_itr, 'loss=%.3f' % epoch_cost, 'train time=%.2f' % epoch_train_time]

#             # self.train_model_help(epoch_itr)

#             ## evaluation
#             if (epoch_itr >= test_from and epoch_itr % test_step == 0) or epoch_itr == num_epochs:
#                 self.eval()
#                 # evaluate model
#                 epoch_eval_start = time()

#                 # test_score = evaluator.evaluate(self)
#                 test_score = evaluator.evaluate_vali(self)
#                 # test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]
#                 #
#                 # print(test_score_str)
#                 updated, should_stop = early_stop.step(test_score, epoch_itr)

#                 # test_score_output = evaluator.evaluate(self)

#                 test_score_output, ndcg_test_all = evaluator.evaluate_full_boost(self)
#                 test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output if k.startswith('NDCG')]
#                 # test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output]

#                 if should_stop:
#                     logger.info('Early stop triggered.')
#                     break
#                 else:
#                     # save best parameters
#                     if updated:
#                         # torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))
#                         # if self.anneal_cap == 1: print(self.anneal)
#                         best_result = test_score_output

#                         # ndcg_test_all = evaluator.evaluate(self, mean=False)

#                         # print(ndcg_test_all['DG@20'])
#                         # print(mean(ndcg_test_all['DG@20']))
#                         # similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name,
#                         #                               'mainstream_scores')
#                         self.make_records()
#                         similarity_file = os.path.join(similarity_dir, 'MF_scores')
#                         if not os.path.exists(similarity_file):
#                             os.mkdir(similarity_file)
#                         with open(os.path.join(similarity_file, self.time + '_mf_scores.npy'), 'wb') as f:
#                             np.save(f, ndcg_test_all)

#                 epoch_eval_time = time() - epoch_eval_start
#                 epoch_time = epoch_train_time + epoch_eval_time

#                 epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
#                 epoch_info += test_score_str
#             else:
#                 epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

#             if epoch_itr % print_step == 0:
#                 logger.info(', '.join(epoch_info))

#             total_train_time = time() - start

#         return best_result, total_train_time

#     # def train_model_help(self, itr):
#     #     epoch_cost = 0.
#     #     epoch_cost1 = 0.
#     #     epoch_cost2 = 0.
#     #     self.user_list, self.item_list, self.label_list = MP_Utility.negative_sampling(self.num_users, self.num_items,
#     #                                                                                 self.train_df[0],
#     #                                                                                 self.train_df[1],
#     #                                                                                 self.neg_sample_rate)
#     #
#     #
#     #     start_time = time() * 1000.0
#     #     num_batch = int(len(self.user_list) / float(self.batch_size)) + 1
#     #     random_idx = np.random.permutation(len(self.user_list))
#     #
#     #     for i in tqdm(range(num_batch)):
#     #         #             print(i)
#     #         batch_idx = None
#     #         if i == num_batch - 1:
#     #             batch_idx = random_idx[i * self.batch_size:]
#     #         elif i < num_batch - 1:
#     #             batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]
#     #
#     #         tmp_cost, tmp_cost1, tmp_cost2 = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
#     #                                                           self.label_list[batch_idx])
#     #
#     #         epoch_cost += tmp_cost
#     #         epoch_cost1 += tmp_cost1
#     #         epoch_cost2 += tmp_cost2
#     #
#     #     print("Training //", "Epoch %d //" % itr, " Total cost = {:.5f}".format(epoch_cost),
#     #           " Total cost1 = {:.5f}".format(epoch_cost1), " Total cost2 = {:.5f}".format(epoch_cost2),
#     #           "Training time : %d ms" % (time() * 1000.0 - start_time))

#     def train_batch(self, user_input, item_input, label_input):
#         # reset gradients
#         self.optimizer.zero_grad()
#         users = torch.Tensor(user_input).int().to(self.device)
#         items = torch.Tensor(item_input).int().to(self.device)
#         labels = torch.Tensor(label_input).float().to(self.device)
#         total_loss = 0
#         # total_loss1 = 0
#         # total_loss2 = 0
#         #
#         # y_hat = self.forward(users, items)
#         # loss = F.mse_loss(y_hat, labels)
#         # # self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() +
#         # #                                                   LA.norm(self.item_factors.weight.data, 'fro').item())
#         #
#         # added_loss = loss.item() + self.regularization_term
#         #
#         # total_loss += added_loss
#         # total_loss1 += loss.item()
#         # total_loss2 += self.regularization_term
#         self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() + LA.norm(self.item_factors.weight.data, 'fro').item())
#         y_hat = self.forward(users, items)
#         # loss = F.mse_loss(y_hat, labels)

#         # loss = self.loss_function(y_hat, labels)
#         y_hat_sig = F.sigmoid(y_hat)
#         loss = -(labels * torch.log(y_hat_sig) + (1 - labels) * torch.log(1 - y_hat_sig)).sum()
#         added_loss = loss + self.regularization_term
#         # loss = self.loss_function(torch.nn.functional.sigmoid(y_hat), labels)
#         # added_loss = loss.item() + self.regularization_term

#         total_loss += added_loss
#         # backpropagate
#         loss.backward()
#         # update
#         self.optimizer.step()

#         # self.eval()

#         # return (total_loss, total_loss1, total_loss2)
#         return total_loss

#     def get_rec(self):
#         P, Q = self.user_factors.weight, self.item_factors.weight
#         P = P.detach().cpu().numpy()
#         Q = Q.detach().cpu().numpy()
#         Rec = np.matmul(P, Q.T)
#         return Rec

#     def make_records(self):  # record all the results' details into files
#         P, Q = self.user_factors.weight, self.item_factors.weight
#         P = P.detach().cpu().numpy()
#         Q = Q.detach().cpu().numpy()
#         similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
#         if not os.path.exists(similarity_dir):
#             os.mkdir(similarity_dir)
#         similarity_file = os.path.join(similarity_dir, 'PC_mf_bce_saves')
#         if not os.path.exists(similarity_file):
#             os.mkdir(similarity_file)
#         with open(os.path.join(similarity_file,'P_MF.npy'), 'wb') as f:
#             np.save(f, P)
#         with open(os.path.join(similarity_file,'Q_MF.npy'), 'wb') as f:
#             np.save(f, Q)
#         # return P, Q

#     def predict(self, user_ids, eval_pos_matrix, eval_items=None):
#         self.eval()
#         batch_eval_pos = eval_pos_matrix[user_ids]
#         with torch.no_grad():
#             # eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
#             # P = torch.Tensor(self.user_list[user_ids]).int()
#             # Q = torch.Tensor(self.item_list[user_ids]).int()
#             Rec = self.get_rec()
#             eval_output = Rec[user_ids, :]
#             if eval_items is not None:
#                 eval_output[np.logical_not(eval_items)] = float('-inf')
#             else:
#                 eval_output[batch_eval_pos.nonzero()] = float('-inf')
#         self.train()
#         return eval_output