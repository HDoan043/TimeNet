import os
import torch
import optuna
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, KANAD, MSGNet, TimeFilter, TimesNet_update_v1, TimesNet_update_v2, LSTMAE


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'LSTMAE': LSTMAE,
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'KANAD': KANAD,
            'MSGNet': MSGNet,
            'TimeFilter': TimeFilter,
            'TimesNetv1': TimesNet_update_v1,
            'TimesNetv2': TimesNet_update_v2
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        trainable_params = sum( p.numel() for p in self.model.parameters() if p.requires_grad)
        print()
        print("="*50)
        print("_ Number of parameters: {}".format(trainable_params))
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
        
    def predict(self):
        pass

    def tune(self):
        if self.args.model != "TimesNet": 
            print("[⚠️] Cannot tune model not being TimesNet!!")
            return

        import gc # Cần import thêm thư viện giải phóng bộ nhớ

        def objective(trial):
            # 1. Gợi ý tham số
            config = {
                'top_k': trial.suggest_int('top_k', 1, 5),
                'num_kernels': trial.suggest_int('num_kernels', 2, 5),
                'd_model': trial.suggest_categorical('d_model', [32, 64, 128, 256]),
                'd_ff': trial.suggest_categorical('d_ff', [256, 512]),
                'e_layers': trial.suggest_int('e_layers', 2, 3),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'dropout': trial.suggest_float('dropout', 0.1, 0.4),
                'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64])
            }

            # Cập nhật tham số vào args
            setting_list = []
            for key, value in config.items():
                setattr(self.args, key, value)
                setting_list.append(f"{key}={value}")
            setting = ",".join(setting_list)

            # --- SỬA LỖI 1: Khởi tạo lại model mới và gán vào self.model ---
            # Trước khi build, xóa model cũ nếu có để giải phóng VRAM
            if hasattr(self, 'model'):
                del self.model
                torch.cuda.empty_cache()
                gc.collect()

            try:
                # Xây dựng model mới dựa trên tham số vừa suggest
                self.model = self._build_model().to(self.device)
                
                # In số lượng tham số để kiểm soát độ nặng
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"\n[Trial {trial.number}] Build model: {trainable_params} params")

                # 2. Huấn luyện (nhớ truyền trial để hỗ trợ pruning trong hàm train)
                self.train(setting=setting, trial=trial)
                
                # 3. Đánh giá (hàm test của bạn trả về 5 giá trị)
                acc, pre, re, f1, threshold = self.test(setting=setting)
                
                print("-" * 80)
                formatted_setting = "\n".join(["\t- " + s for s in setting.split(",")])
                print(f"Results for Trial {trial.number}:\n{formatted_setting}")
                print(f"> Accuracy: {acc:.4f}, Precision: {pre:.4f}, Recall: {re:.4f}, F-score: {f1:.4f} | threshold: {threshold}")
                
                return f1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ [Trial {trial.number}] OUT OF MEMORY - Pruning this configuration.")
                    # --- SỬA LỖI 2: Dọn dẹp sạch sẽ trước khi thoát trial ---
                    if hasattr(self, 'model'):
                        del self.model
                    torch.cuda.empty_cache()
                    gc.collect()
                    raise optuna.exceptions.TrialPruned()
                else:
                    raise e

        # Create and run study
        study_name = "timesnet_max_tuning"
        db_path = os.path.join(self.args.tune_path, "timesnet_optuna.db")
        if not os.path.exists(db_path):
            print("⚠️⚠️⚠️[NEW STUDY] Starting a new study of optuna. If you want to run a saved study, please pass the right path")
        os.makedirs(self.args.tune_path, exist_ok = True)
        storage_url = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler()
        )
        study.optimize(objective, n_trials=self.args.num_trials, timeout=11*3600)
        
        # In kết quả tốt nhất
        print("\n" + "="*50)
        print("🏆 BEST CONFIGURATION FOUND:")
        print(f"  Best F1: {study.best_trial.value:.4f}")
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
