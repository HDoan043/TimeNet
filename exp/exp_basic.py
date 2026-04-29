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
        # Define objective
        def objective(trial):
            if self.args.model != "TimesNet": 
                print("[⚠️] Cannot tune model not being TimesNet!!")
                return
            config = {
                'top_k': trial.suggest_int('top_k', 1, 5),
                'num_kernels': trial.suggest_int('num_kernels', 2, 7),
                'd_model': trial.suggest_categorical('d_model', [64, 128, 256, 512]),
                'd_ff': trial.suggest_int('d_ff', [256, 512, 1024, 2048]),
                'e_layers': trial.suggest_int('e_layers', 2, 3),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'anomaly_ratio': trial.suggest_float('anomaly_ratio', 6.0, 13.0)
            }
            setting = ""
            for i, (key, value) in enumerate(config.items()):
                if i != 0:
                    setting = setting + ","
                self.args[key] = value
                setting = setting + f"{key}={value}"
            self._build_model()
            self.train(setting = setting, trial = trial)
            acc, pre, re, f1, threshold = self.test(setting = setting)
            print("-"*80)
            setting = setting.split(",")
            setting = "\n".join(["\t- "+each for each in setting])
            print(setting)
            print(f"> Accuracy: {acc}, Precision: {pre}, Recall: {re}, F-score: {f1}| threshold: {threshold}")
            print ()
            return f1
        
        # Create and run study
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=self.args.n_trials)
        
        # Print best result
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value (F1): {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
