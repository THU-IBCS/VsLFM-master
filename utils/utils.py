import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import numpy as np
import logging
import torch
import tifffile as tiff


class TrainSetLoader(Dataset):      # Preprossesing of training data
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir + 'train_LR/')
        item_num = len(file_list)   # Get the number of images in the dataset.
        max_value = 3000            # The maxvalue used for normalization. You can adjust it depend on the dataset used.

        self.item_num = item_num
        self.max_value =max_value


    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        HR_file_name = [dataset_dir + 'train_HR' + '/%05d' % index + 'HR.tif']
        LR_file_name = [dataset_dir + 'train_LR' + '/%05d' % index + 'LR.tif']

        # Normalization
        label = tiff.imread(HR_file_name)/self.max_value
        data = tiff.imread(LR_file_name)/self.max_value


        return torch.from_numpy(data.copy()).type(torch.FloatTensor), torch.from_numpy(label.copy()).type(torch.FloatTensor)

    def __len__(self):
        return self.item_num



def get_logger(filename, verbosity=1, name=None):  # Log information
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

