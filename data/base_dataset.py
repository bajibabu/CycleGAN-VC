import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_setter(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

    def __len__(self):
        return 0


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'none':
        transform_list.append(transforms.Lambda(
            lambda img: __adjust(img)))
    else:
        raise ValueError('--resize_or_crop % is not a valid option' % opt.resize_or_crop)
    
    #transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)

# just modify the width and height to be multiple of 4
def __adjust(img):
    oh, ow = img.shape

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    if ow % mult == 0 and oh % mult == 0:
        # convert to torch tensor and add channel dimension
        return torch.unsqueeze(torch.from_numpy(img), 0)
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)
    
    new_img = np.zeros((h, w), dtype=np.float32)
    new_img[:oh,:ow] = img

    return torch.unsqueeze(torch.from_numpy(new_img), 0) # TODO: remove hard code here


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). The adjustment will be done to all images "
              "whose sizes are not multiple of 4" %(ow, oh, w, h))
        __print_size_warning.has_printed = True