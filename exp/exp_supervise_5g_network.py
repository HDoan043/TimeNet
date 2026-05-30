from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment, ProgressBar, aggregate
from utils.pate_metric import PATE
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, average_precision_score
import torch.multiprocessing
from utils.losses import *

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
import json

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings('ignore')


class Exp_Supervise_5G_Network(Exp_Basic):
    def __init__(self, args):
        super(Exp_Supervise_5G_Network, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
    
        return model

    def _get_data(self, flag, phase='test'):
        data_set, data_loader = data_provider(self.args, flag, phase=phase)
        return data_set, data_loader

    def _select_optimizer(self):
        real_model = (self.model.module if isinstance(self.model, nn.DataParallel) else self.model)
        
        # LUÔN LUÔN chỉ đưa vào Optimizer những parameter đang mở khóa
        active_params = filter(lambda p: p.requires_grad, real_model.parameters())
        
        model_optim = optim.Adam(active_params, lr=self.args.learning_rate)
            
        return model_optim
        
    def _select_criterion(self):
        if self.args.corrector_criterion.lower() in ["correctorbceloss", "correctorbce", "bce", "binary_cross_entropy_loss", "binary cross entropy"]:
            criterion = CorrectorBCELoss(self.args)
        else:
            criterion = CorrectorMIL_Loss(self.args)
        return criterion
      
    def train(self, setting, trial=None):
        train_data, train_loader = self._get_data(flag='train', phase="train")
        raw_train_data, raw_train_loader = self._get_data(flag='train', phase="test")
        vali_data, vali_loader = self._get_data(flag='test', phase="test")

        if self.args.from_pretrained != "":
            print(f'[⏯️]Training from pretrained model ...')
            checkpoint_path = os.path.join(self.args.from_pretrained, setting, 'checkpoint.pth')
            backup_checkpoint_path = os.path.join(self.args.from_pretrained, 'checkpoint.pth')
            backup_checkpoint_path2= self.args.from_pretrained
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
            elif os.path.exists(backup_checkpoint_path):
                self.model.load_state_dict(torch.load(backup_checkpoint_path), strict=False)
            elif os.path.exists(backup_checkpoint_path2):
                self.model.load_state_dict(torch.load(backup_checkpoint_path2), strict=False)
            else:
                print(f'[⚠️] Cannot find the pretrained model, start training from 0...')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        time_begin= time.time()
        aggregate_steps = 0

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        best_epoch = 0
        best_pate = 0
        best_roc_auc = 0
        best_pr_auc = 0
        best_f1 = 0
        best_acc = 0
        best_pre = 0
        best_rec = 0
        best_threshold = 0
        best_result = (0,0,0,0,0)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            print()
            print("="*25 + " Epoch [{}]: ".format(epoch+1)+"="*25)
            print()
            
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            pbar = ProgressBar(train_loader, bin=60)
            i = 0

            for batch in train_loader:
            # for batch in pbar:
                aggregate_steps += 1
                iter_count += 1
                model_optim.zero_grad()
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.to(self.device) 
                batch_y = batch_y.float().to(self.device).squeeze(-1)

                outputs = self.model(batch_x, batch_x_mark, None, None)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
        
                speed = (time.time() - time_begin) / aggregate_steps
                left_time_s = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                if left_time_s <60: 
                    left_time = f"{round(left_time_s,4)}s"
                elif left_time_s<3600:
                    left_time = f"{round(left_time_s/60,4)}mins"
                else: left_time = f"{round(left_time_s/3600,4)}hs"
    
                # pbar.set_postfix(
                #     {
                #         "Epoch": epoch + 1,
                #         "Iteration": f"{i+1}/{train_steps}",
                #         "Loss": loss.item(),
                #         "Speed": f"{round(speed, 4)}s/iter",
                #         "Left time": left_time
                #     }
                # )
        
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

                i+=1
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            train_loss = np.average(train_loss)
            result = self.test(setting, train_load=(raw_train_data, raw_train_loader), test_load=(vali_data, vali_loader))
            if len(result)>1:
                vali_best_acc, vali_best_pre, vali_best_rec, vali_best_f1, vali_best_threshold = result
                if vali_best_f1 >= best_f1:
                    best_acc = vali_best_acc
                    best_pre = vali_best_pre 
                    best_rec = vali_best_rec
                    best_f1 = vali_best_f1
                    best_epoch = epoch
                    best_threshold = vali_best_threshold
                    best_result = (best_acc, best_pre, best_rec, best_f1, best_threshold)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali best f1: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_best_f1))
                score = vali_best_f1
                
            else: 
                if self.args.metric.lower() == "pate":
                    pate_score = result[0]
                    if pate_score >= best_pate:
                        best_epoch = epoch
                        best_pate = pate_score
                        best_result = (0,0,0,best_pate,0)
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Pate best: {3:.7f}".format(
                        epoch + 1, train_steps, train_loss, pate_score))
                    score = pate_score
                elif self.args.metric.lower() in ["roc_auc", "roc-auc", "rocauc", "roc"]:
                    roc_score = result[0]
                    if roc_score >= best_roc_auc:
                        best_epoch = epoch
                        best_roc_auc = roc_score
                        best_result = (0,0,0,best_roc_auc,0)
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} ROC-AUC best: {3:.7f}".format(
                        epoch + 1, train_steps, train_loss, roc_score))
                    score = roc_score
                else:
                    pr_auc_score = result[0]
                    if pr_auc_score >= best_pr_auc:
                        best_epoch = epoch
                        best_pr_auc = pr_auc_score
                        best_result = (0,0,0,best_pr_auc,0)
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} PR-AUC best: {3:.7f}".format(
                        epoch + 1, train_steps, train_loss, pr_auc_score))
                    score = pr_auc_score
            early_stopping(score, self.model, path)
            if trial:
                trial.report(score, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                    
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return best_result

    def test(self, setting, test=0, train_load=None, test_load=None):
        if train_load:
            train_data, train_loader = train_load
        else:
            train_data, train_loader = self._get_data(flag='train', phase="test")
        if test_load:
            test_data, test_loader = test_load
        else:
            test_data, test_loader = self._get_data(flag='test', phase="test")
        
        timestamps = test_data.get_timestamps()

        inference_times = []
        
        if test:
            print('loading model')
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
            else:
                backup_checkpoint_path = os.path.join(self.args.checkpoints, 'checkpoint.pth')
                self.model.load_state_dict(torch.load(backup_checkpoint_path), strict=False)

        attens_energy = []
        folder_path = self.args.test_result_path if self.args.test_result_path != "" else './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        timestamps.to_csv(os.path.join(folder_path, "timestamps.csv"))
            
        self.model.eval()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        # (1) stastic on the train set
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                (batch_x, batch_y, batch_x_mark, batch_y_mark) = batch
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.to(self.device) 
                score = self.model(batch_x, batch_x_mark, None, None)    # [B, win_size]
                prob_score = torch.sigmoid(score) 
                score_np = prob_score.detach().cpu().numpy()
                attens_energy.append(score_np)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)            # attens_energy: [num_batch * batch_size * win_size]
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        timestamps = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.to(self.device) 
                # reconstruction
                if self.args.use_gpu:
                    torch.cuda.synchronize()
                start_time = time.time()
                score = self.model(batch_x, batch_x_mark, None, None)
          
                if self.args.use_gpu:
                    torch.cuda.synchronize() # Đợi GPU chạy xong 100%
                end_time = time.time()
    
                inference_times.append((end_time - start_time) * 1000)            
                prob_score = torch.sigmoid(score) 
                score_np = prob_score.detach().cpu().numpy()
                attens_energy.append(score_np)                                            # attens_energy: [batch_size x win_size] x num_batch
                if i > len(test_loader) -2:
                    print("score: {}".format(score.shape))
                    print("batch_y: {}".format(batch_y.shape))
                test_labels.append(batch_y)
                batch_stamps = batch_x_mark.detach().cpu().numpy()
                batch_stamps = [test_data.decode_timestamp(data_timestamp) for data_timestamp in batch_stamps]
                timestamps.append(batch_stamps)
            
        attens_energy = np.concatenate(attens_energy, axis=0)                        # attens_energy: [batch_size*num_batch x win_size]
        ######################################
        # Save predict result
        np.save(folder_path + "pred.npy", attens_energy)
        ######################################
        test_energy = np.array(attens_energy.reshape(-1))                            # test_energy: [batch_size * num_batch * win_size]
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        test_labels = np.concatenate(test_labels, axis=0)
        
        test_labels = np.array(test_labels.reshape(-1))
        gt = test_labels.astype(int)

        timestamps = np.concatenate(timestamps, axis=0)
        timestamps = np.array(timestamps.reshape(-1))
        timestamps = pd.to_datetime(timestamps, format='%Y-%m-%d %H:%M:%S')
        
        predict_df = pd.DataFrame({"date": timestamps, "score": test_energy, "label": gt})
        
        predict_df = aggregate(predict_df, self.args.aggregate)
        predict_df.to_csv(folder_path + "anomaly_score_df.csv")
        gt = predict_df["label"].values
        test_energy = predict_df["score"].values
        # pate_score = PATE(gt, test_energy, e_buffer=6, d_buffer=12, binary_scores=False)
        roc_auc = roc_auc_score(gt, test_energy)
        pr_auc = average_precision_score(gt, test_energy)
        # print(f"PATE score: {pate_score}")
        print(f"ROC-AUC score: {roc_auc}")
        print(f"PR-AUC score: {pr_auc}")

        if self.args.threshold > -1:
            threshold = self.args.threshold
            print("Use provided threshold :", threshold)
                
            # (3) evaluation on the test set
            pred = (test_energy > threshold).astype(int)
            ######################################
            # Save ground truth
            np.save(folder_path + "true.npy", test_labels)
            ######################################
            
            # (4) detection adjustment
            gt, pred = adjustment(gt, pred)
            pred = np.array(pred)
            gt = np.array(gt)
            
            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
            print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
            # Calculate batch time
            avg_time_ms = np.mean(inference_times)
            std_time_ms = np.std(inference_times) 
        
            print(f"Mean batch times: {avg_time_ms:.2f} ms ± {std_time_ms:.2f} ms")
            max_memory_bytes = torch.cuda.max_memory_allocated(self.device)
            max_memory_mb = max_memory_bytes / (1024 * 1024)
            print(f"Peak Memory: {max_memory_mb:.2f} MB")
    
            f = open("result_anomaly_detection.txt", 'a')
            f.write(setting + "  \n")
            f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
            f.write('\n')
            f.write('\n')
            f.close()
        else:
            print(f"=== TRYING ANOMALY RATIOS {self.args.anomaly_ratio} ===")
            best_ratio = 0
            best_acc = 0
            best_pre = 0
            best_re = 0
            best_f1 = 0
            
            for each in self.args.anomaly_ratio:
                threshold = np.percentile(combined_energy, 100 - each)
                    
                # (3) evaluation on the test set
                pred = (test_energy > threshold).astype(int)
                
        
                # (4) detection adjustment
                gt, pred = adjustment(gt, pred)
        
                pred = np.array(pred)
                gt = np.array(gt)
            
                accuracy = accuracy_score(gt, pred)
                precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
                # print("\tAccuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                #     accuracy, precision,
                #     recall, f_score))
                if f_score >= best_f1:
                    best_acc = accuracy
                    best_pre = precision
                    best_re = recall
                    best_f1 = f_score
                    best_ratio = each
                    best_threshold = threshold
                # Calculate batch time
                avg_time_ms = np.mean(inference_times)
                std_time_ms = np.std(inference_times) 
    
                max_memory_bytes = torch.cuda.max_memory_allocated(self.device)
                max_memory_mb = max_memory_bytes / (1024 * 1024)
    
            print(f"Best anomaly_ratio: {best_ratio}, Best threshold: {best_threshold}")
            print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                best_acc, best_pre, best_re, best_f1))
            accuracy = best_acc
            precision = best_pre
            recall = best_re
            f_score = best_f1
            threshold = best_threshold
        # if self.args.metric.lower() == "pate": return [pate_score]
        if self.args.metric.lower() in ["roc_auc", "roc-auc", "roc", "rocauc"]: return [roc_auc]
        elif self.args.metric.lower() in ["pr_auc", "pr-auc", "prauc", "pr", "average precision score"]: return [pr_auc]
        else: return accuracy, precision, recall, f_score, threshold
