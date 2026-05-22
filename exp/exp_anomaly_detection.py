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


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # 1. Trích xuất model gốc từ DataParallel (Giữ nguyên logic đúng của bạn)
        real_model = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
        
        freeze_layers = self.args.freeze
        
        if freeze_layers and len(freeze_layers) > 0:
            print(f"[☃️] Freezing layers matching: {freeze_layers}...")
            
            # Tập hợp các layer đã tìm thấy để kiểm tra layer nào bị viết sai chính tả
            matched_layers = set()
            
            # Duyệt qua từng module con trong mô hình
            for module_name, module in real_model.named_modules():
                # Kiểm tra xem module này có nằm trong danh sách cần freeze không
                for target_layer in freeze_layers:
                    # So khớp chính xác hoặc khớp lớp con (ví dụ: 'layers.0' sẽ khớp với 'layers.0.conv')
                    if module_name == target_layer or module_name.startswith(target_layer + "."):
                        matched_layers.add(target_layer)
                        
                        # Freeze toàn bộ parameter của module này
                        for param in module.parameters():
                            param.requires_grad = False
        
            # Cảnh báo nếu người dùng nhập sai tên layer từ môi trường ngoài
            for target_layer in freeze_layers:
                if target_layer not in matched_layers:
                    print(f"[⚠️] Warning: Model has no module named '{target_layer}'")
    
        return model

    def _get_data(self, flag, contrastive=False):
        data_set, data_loader = data_provider(self.args, flag, contrastive=contrastive)
        return data_set, data_loader

    def _select_optimizer(self):
        real_model = ( self.model.module if isinstance(self.model, nn.DataParallel) else self.model)
        if self.args.contrastive == 1 and self.args.is_training == 1:
            base_params = []
            projector_params = list(
                real_model.contrastive_project.parameters()
            )
            projector_param_ids = {
                id(p) for p in projector_params
            }
    
            for p in real_model.parameters():
                if id(p) not in projector_param_ids:
                    base_params.append(p)
    
            model_optim = optim.AdamW([
                {
                    "params": base_params,
                    "lr": self.args.learning_rate
                },
                {
                    "params": projector_params,
                    "lr": self.args.learning_rate * 100
                }
            ])
    
        else:
            model_optim = optim.Adam(
                real_model.parameters(),
                lr=self.args.learning_rate
            )
    
        return model_optim

    def _select_criterion(self):
        if self.args.contrastive == 1 and self.args.is_training == 1:
            if self.args.contrastive_criterion.lower() in ["ntxent", "nt-xent", "nt_xent", "nt_xent_loss", "nt-xent_loss"]:
                criterion = NTXentLoss(self.args)
            elif self.args.contrastive_criterion.lower() in ["triplet", "triplet_loss", "triplet-loss"]:
                criterion = TripletLoss(self.args)
            else:
                criterion = SeSimiLoss(self.args)
        else: criterion = nn.MSELoss()
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
                self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
            elif os.path.exists(backup_checkpoint_path):
                self.model.load_state_dict(torch.load(backup_checkpoint_path), strict=False)
            elif os.path.exist(backup_checkpoint_path2):
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
            epoch_logs = {"loss_recon": 0, "loss_contrastive": 0, "recon_gap": 0, "throttle": 0, "alpha_sim": 0, "alpha_mag": 0}
            # for batch in train_loader:
            for batch in pbar:
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
                    batch_all_samples, batch_all_mark, batch_base_mse, idx, pos_idx, neg_idx, label = batch
                    batch_x = batch_all_samples.float().to(self.device)
                    batch_x_mark = batch_all_mark.to(self.device)
                    
                    hidden_state, outputs, attn_pooling = self.model(batch_x, batch_x_mark, None, None)
                    loss, metrics = criterion(batch_x, outputs, hidden_state, idx, pos_idx, neg_idx, label, attn_pooling, batch_base_mse)
                    train_loss.append(loss.item())

                    for key in epoch_logs.keys():
                        epoch_logs[key] += metrics[key]

                speed = (time.time() - time_begin) / aggregate_steps
                left_time_s = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                if left_time_s <60: 
                    left_time = f"{round(left_time_s,4)}s"
                elif left_time_s<3600:
                    left_time = f"{round(left_time_s/60,4)}mins"
                else: left_time = f"{round(left_time_s/3600,4)}hs"
    
                pbar.set_postfix(
                    {
                        "Epoch": epoch + 1,
                        "Iteration": f"{i+1}/{train_steps}",
                        "Loss": loss.item(),
                        "Speed": f"{round(speed, 4)}s/iter",
                        "Left time": left_time
                    }
                )
                # Sau khi kết thúc vòng lặp Batch (Hết 1 Epoch):
        
                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                loss.backward()
                t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

                i+=1
            
            num_batches = len(train_loader)
            for key in epoch_logs.keys():
                epoch_logs[key] /= num_batches # Tính trung bình cả Epoch
            
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # In ra màn hình hoặc ghi vào file log:
            print(f"\nEpoch {epoch} | Recon: {epoch_logs['loss_recon']:.4f} | Cont: {epoch_logs['loss_contrastive']:.4f} | Gap: {epoch_logs['recon_gap']:.4f} | Throttle: {epoch_logs['throttle']:.4f}")
            
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, corr_matrix, criterion)
            result = self.test(setting)
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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train', contrastive=False)
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
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        # (1) stastic on the train set
        save_train_loss = []
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                (batch_x, batch_y, batch_x_mark, batch_y_mark) = batch
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.to(self.device) 
                # reconstruction
                if self.args.contrastive == 0:
                    outputs = self.model(batch_x, batch_x_mark, None, None)    
                else:
                    hidden_state, outputs, attn_pooling = self.model(batch_x, batch_x_mark, None, None, None)
                # criterion

                score = self.anomaly_criterion(batch_x, outputs)        # [B, win_size, channels]
                win_score = score.mean(dim=(1,2)).detach().cpu().numpy()# [B]
                save_train_loss.append(win_score)
                score = torch.mean(score, dim=-1)                       # [B, win_size]
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)            # attens_energy: [num_batch * batch_size * win_size]
        train_energy = np.array(attens_energy)
        save_train_loss = np.array(save_train_loss)                    # [B] x num_batch
        save_train_loss = np.concatenate(save_train_loss, axis=0)      # [B * num_batch]
        np.save(folder_path + "train_reconstruct_loss.npy", save_train_loss)

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
                if self.args.contrastive == 0:
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                else:
                    hidden_state, outputs, attn_pooling = self.model(batch_x, batch_x_mark, None, None, None)
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
