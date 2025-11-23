import torch
import numpy as np
import os

class EarlyStopping:
    """
    早停工具，用于在验证集指标不再提升时停止训练，并保存最佳模型。
    """
    def __init__(self, save_dir, patience=10, verbose=False, save_model_name='best_model.pt'):

        self.save_path = os.path.join(save_dir, save_model_name)
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_max = -np.Inf

    def __call__(self, val_metric, model):


        score = val_metric

        if self.best_score is None:

            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score:

            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping 计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:

            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):

        if self.verbose:
            print(f'验证指标提升 ({self.val_metric_max:.6f} --> {val_metric:.6f})。 保存模型到 {self.save_path} ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_metric_max = val_metric