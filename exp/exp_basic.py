import os
import torch
from models import (
    Autoformer,
    Transformer,
    TimesNet,
    Nonstationary_Transformer,
    DLinear,
    Informer,
    LightTS,
    Reformer,
    PatchTST,
    Crossformer,
    FiLM,
    iTransformer,
    TiDE,
    TimeMixer,
    TSMixer,
    MambaSimple,
)

try:  # Optional dependency for the main model
    from models import ICMamba
    _has_icmamba = True
except Exception as _e:
    ICMamba = None  # type: ignore
    _has_icmamba = False
    _icmamba_import_error = _e


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'PatchTST': PatchTST,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'TiDE': TiDE,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
        }
        if _has_icmamba:
            self.model_dict['ICMamba'] = ICMamba
        else:
            print(f"Warning: ICMamba is unavailable ({_icmamba_import_error})")
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
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
