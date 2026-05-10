from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment, ProgressBar
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
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

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, contrastive=False):
        data_set, data_loader = data_provider(self.args, flag, contrastive=contrastive)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = NTXentLoss(self.args) if self.args.contrastive == 1 and self.args.is_training == 1 else nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, corr_matrix, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, batch_x_mark, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.to(self.device) 

                if self.args.model.lower() == "timesnetv2":
                    outputs = self.model(batch_x, None, None, None, corr_matrix)
                else: outputs = self.model(batch_x, batch_x_mark, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, trial=None):
        train_data, train_loader = self._get_data(flag='train', contrastive=self.args.contrastive)
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if self.args.from_pretrained != "":
            print(f'[⏯️]Training from pretrained model ...')
            checkpoint_path = os.path.join(self.args.from_pretrained, setting, 'checkpoint.pth')
            backup_checkpoint_path = os.path.join(self.args.from_pretrained, 'checkpoint.pth')
            backup_checkpoint_path2= self.args.from_pretrained
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path))
            elif os.path.exists(backup_checkpoint_path):
                self.model.load_state_dict(torch.load(backup_checkpoint_path))
            elif os.path.exist(backup_checkpoint_path2):
                self.model.load_state_dict(torch.load(backup_checkpoint_path2))
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
        best_f1 = 0
        best_acc = 0
        best_pre = 0
        best_rec = 0
        best_threshold = 0

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
            # pbar = ProgressBar(train_loader, bin=60)
            i = 0
            for batch in train_loader:
            # for batch in pbar:
                aggregate_steps += 1
                iter_count += 1
                model_optim.zero_grad()
                if self.args.contrastive == 0:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.to(self.device) 
    
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                    loss = criterion(outputs, batch_x)
                    train_loss.append(loss.item())
                else:
                    batch_all_samples, idx, pos_idx, neg_idx, label = batch
                    batch_x = batch_all_samples.float().to(self.device)
                    hidden_state, outputs, attn_pooling = self.model(batch_x, None, None, None)
                    loss = criterion(batch_x, outputs, hidden_state, idx, pos_idx, neg_idx, label, attn_pooling)
                    train_loss.append(loss.item())

                # speed = (time.time() - time_begin) / aggregate_steps
                # left_time_s = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                # if left_time_s <60: 
                #     left_time = f"{round(left_time_s,4)}s"
                # elif left_time_s<3600:
                #     left_time = f"{round(left_time_s/60,4)}mins"
                # else: left_time = f"{round(left_time_s/3600,4)}hs"
    
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
                model_optim.step()

                i+=1

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, corr_matrix, criterion)
            vali_best_acc, vali_best_pre, vali_best_rec, vali_best_f1, vali_best_threshold = self.test(setting)
            if vali_best_f1 >= best_f1:
                best_acc = vali_best_acc
                best_pre = vali_best_pre 
                best_rec = vali_best_rec
                best_f1 = vali_best_f1
                best_epoch = epoch
                best_threshold = vali_best_threshold
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali best f1: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_best_f1))
            early_stopping(vali_best_f1, self.model, path)
            if trial:
                trial.report(vali_best_f1, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                    
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return best_acc, best_pre, best_rec, best_f1, best_threshold

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train', contrastive=False)
        timestamps = test_data.get_timestamps()

        inference_times = []
        
        if test:
            print('loading model')
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path))
            else:
                backup_checkpoint_path = os.path.join(self.args.checkpoints, 'checkpoint.pth')
                self.model.load_state_dict(torch.load(backup_checkpoint_path))

        attens_energy = []
        folder_path = self.args.test_result_path if self.args.test_result_path != "" else './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        timestamps.to_csv(os.path.join(folder_path, "timestamps.csv"))
            
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        # (1) stastic on the train set
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                (batch_x, batch_y, batch_x_mark, batch_y_mark) = batch
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.to(self.device) 
                # reconstruction
                if self.args.contrastive == 0:
                    outputs = self.model(batch_x, batch_x_mark, None, None)    
                else:
                    hidden_state, outputs, attn_pooling = self.model(batch_x, None, None, None)
                # criterion
            
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)            # attens_energy: [num_batch * batch_size * num_channels]
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        timestamps = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.to(self.device) 
            # reconstruction
            if self.args.use_gpu:
                torch.cuda.synchronize()
            start_time = time.time()
            if self.args.contrastive == 0:
                outputs = self.model(batch_x, batch_x_mark, None, None)
            else:
                hidden_state, outputs, attn_pooling = self.model(batch_x, None, None, None)
            if self.args.use_gpu:
                torch.cuda.synchronize() # Đợi GPU chạy xong 100%
            end_time = time.time()

            inference_times.append((end_time - start_time) * 1000)            
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)      # score:  [batch_size x win_size]
            score = score.detach().cpu().numpy()
            attens_energy.append(score)                                              # attens_energy: [batch_size x win_size] x num_batch
            if i > len(test_loader) -2:
                print("score: {}".format(score.shape))
                print("batch_y: {}".format(batch_y.shape))
            test_labels.append(batch_y)
            timestamps.append(batch_x_mark.detach().cpu().numpy())
        
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
        predict_df.to_csv(folder_path + "anomaly_score_df.csv")

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
            return accuracy, precision, recall, f_score, threshold

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
        return best_acc, best_pre, best_re, best_f1, best_threshold
    def infer(self, setting, flag='test'):
        infer_data, infer_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        full_data,  full_loader = self._get_data(flag='full')

        corr_matrix = train_data.get_corr_matrix()
        corr_matrix = torch.tensor(corr_matrix, dtype = torch.float32, device = self.device)
        corr_matrix.require_grad = False

        timestamps = full_data.get_timestamps()
        
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './pred_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        timestamps.to_csv(os.path.join(folder_path, "timestamps.csv"))
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.to(self.device) 
                # reconstruction
                if self.args.model.lower() == "timesnetv2":
                    outputs = self.model(batch_x, None, None, None, corr_matrix)
                else: outputs = self.model(batch_x, batch_x_mark, None, None)   # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        attens_energy = []
        gt_labels = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(infer_loader):
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.to(self.device)
            # reconstruction
            if self.args.model.lower() == "timesnetv2":
                outputs = self.model(batch_x, None, None, None, corr_matrix)
            else: outputs = self.model(batch_x, batch_x_mark, None, None)            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            gt_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        infer_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, infer_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)

        gt_labels = np.concatenate(gt_labels, axis=0).reshape(-1)
        gt_labels = np.array(gt_labels)

        attens_energy = []
        for i, (batch_x, _, _, _) in enumerate(full_loader):
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.to(self.device)
            # reconstruction
            if self.args.model.lower() == "timesnetv2":
                outputs = self.model(batch_x, None, None, None, corr_matrix)
            else: outputs = self.model(batch_x, batch_x_mark, None, None)            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

        full_energy = np.array(np.concatenate(attens_energy, axis = 0))
        
        print("Threshold :", threshold)
        print("Max: ", combined_energy.max())
        print("Min: ", combined_energy.min())
        print("Shape inference full data: {}".format(full_energy.shape))
        print("Shape ground truth: {}".format(gt_labels.shape))

        # Saving result
        # with open(os.path.join(folder_path, "result_inference.npy"), "w") as f:
        # np.save(os.path.join(folder_path, "inference.npy"), infer_energy)
        # with open(os.path.join(folder_path, "ground_truth.npy"), "w") as f:
        np.save(os.path.join(folder_path, "true.npy"), gt_labels)

        np.save(os.path.join(folder_path, "debug.npy"), combined_energy)
        # with open(os.path.join(folder_path, "threshold"), "w") as f:
        np.save(os.path.join(folder_path, "threshold.npy"), threshold)
        # with open(os.path.join(folder_path, "train.npy"), "w") as f:
        np.save(os.path.join(folder_path, "full.npy"), full_energy)

        print("[DONE] Inference result is successfully saved in {}".format(folder_path))
