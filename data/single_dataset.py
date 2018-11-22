import os.path
import numpy as np
from data.base_dataset import BaseDataset, get_transform
from data.mgc_folder import make_dataset
import random

class SingleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
    
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        if opt.direction == 'AtoB':
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')    
            self.A_paths = make_dataset(self.dir_A)
            self.A_paths = sorted(self.A_paths)
            self.length = len(self.A_paths)
        elif opt.direction == 'BtoA':
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
            self.B_paths = make_dataset(self.dir_B)
            self.B_paths = sorted(self.B_paths)
            self.length = len(self.B_paths)
        else:
            raise RuntimeError('Only directions AtoB | BtoA are supported.')

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        if self.opt.direction == 'AtoB':
            A_path = self.A_paths[index]
            A_mgc = np.load(A_path)['mel_spec'].astype(np.float32).T
            A = self.transform(A_mgc)
            input_nc = self.opt.input_nc
            return {'A': A, 'A_paths': A_path}
        elif self.opt.direction == 'BtoA':
            B_path = self.B_paths[index]
            B_mgc = np.load(B_path)['mel_spec'].astype(np.float32).T
            B = self.transform(B_mgc)
            input_nc = self.opt.output_nc
            return {'B': B, 'B_paths': B_path}
        else:
            raise RuntimeError('Only directions AtoB | BtoA are supported.')

    def __len__(self):
        return self.length

    def name(self):
        return 'SingleDataset'