###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads mgc files from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

import os
import os.path

MGC_EXTENSIONS = ['.npz']


def is_mgc_file(filename):
    return any(filename.endswith(extension) for extension in MGC_EXTENSIONS)


def make_dataset(dir):
    mgcs = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_mgc_file(fname):
                path = os.path.join(root, fname)
                mgcs.append(path)

    return mgcs


def default_loader(path):
    return _read_binary_file(path)


class MGCFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        mgcs = make_dataset(root)
        if len(mgcs) == 0:
            raise(RuntimeError("Found 0 mgc files in: " + root + "\n"
                               "Supported mgc extensions are: " +
                               ",".join(MGC_EXTENSIONS)))
        
        self.root = root
        self.mgcs = mgcs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.mgcs[index]
        mgc = self.loader(path)
        if self.transform is not None:
            mgc = self.transform(mgc)
        if self.return_paths:
            return mgc, path
        else:
            return mgc
    
    def __len__(self):
        return len(self.mgcs)