from models.utils import standard_train
from models.basenet import BaseNet
from importlib import import_module
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class baseline(BaseNet):
    def __init__(self, opt, wandb):
        super(baseline, self).__init__(opt, wandb)
        self.set_network(opt)
        self.set_optimizer(opt)

    def set_network(self, opt):
        """Define the network"""
        
        if self.is_3d:
            mod = import_module("models.basemodels_3d")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained = self.pretrained).to(self.device)
        elif self.is_tabular:
            mod = import_module("models.basemodels_mlp")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, in_features= self.in_features, hidden_features = 1024).to(self.device)
        else:
            mod = import_module("models.basemodels")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained=self.pretrained).to(self.device)
            
    def _train(self, loader):
        """Train the model for one epoch"""

        self.network.train()
        auc, train_loss,pred_df = standard_train(self.opt, self.network, self.optimizer, loader, self._criterion, self.wandb, self.scheduler) #  if you don't add scheduler as an arg it doesn't run

        print('Training epoch {}: AUC:{}'.format(self.epoch, auc))
        print('Training epoch {}: loss:{}'.format(self.epoch, train_loss))

        # to distinguish between pretrained and not pretrained model results
        if self.pretrained:
            pred_df.to_csv(os.path.join(self.save_path, 'pretrained_epoch_' + str(self.epoch)+'_train_pred.csv'), index = False)

        else:
            pred_df.to_csv(os.path.join(self.save_path, 'not_pretrained_epoch_' + str(self.epoch)+'_train_pred.csv'), index = False)

        self.epoch += 1
    