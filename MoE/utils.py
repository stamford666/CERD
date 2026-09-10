import datetime
import random
import numpy as np
import os
import torch
import logging

# Set random seed for reproducibility
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(log_path, log_name, file_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.abspath(os.path.join(log_path, file_name))
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file:
            return logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
