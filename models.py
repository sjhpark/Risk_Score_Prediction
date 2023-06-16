# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy

# model 1: RiskScoreModel w/ PyTorch Lightning
class RiskScoreModel_PL(pl.LightningModule):
    def __init__(self, num_features, hidden_size, num_classes, lr):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # layers
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, num_classes)
        self.relu = nn.ReLU() # activation function (no learnable parameters)
        self.dropout = nn.Dropout(0.2) # dropout layer

        # hyperparameters for training
        self.lr = lr # learning rate

        # model architecture
        self.model = nn.Sequential(
            self.fc1,
            self.relu,
            self.dropout,
            self.fc2,
            self.relu,
            self.dropout,
            self.fc3
            )
    
    def forward(self, x):
        '''
        forward function is for making predictions
        '''
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        '''
        training_step is for training the model
        '''
        x, y = batch
        y_logits = self(x) # predictions; self(): calls forward function
        loss = F.cross_entropy(y_logits, y) # loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        '''
        validation_step is for validation
        '''
        x, y = batch
        y_logits = self(x)
        loss = F.cross_entropy(y_logits, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        '''
        test_step is for testing
        '''
        x, y = batch
        y_logits = self(x)
        loss = F.cross_entropy(y_logits, y)
        self.log('test_loss', loss)
        accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(x.device)
        acc = accuracy(y_logits, y)
        self.log('test_accuracy [%]', acc*100)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# model 2: RiskScoreModel + Ensemble Learning with PyTorch
class RiskScoreEnsembleModel(nn.Module):
    def __init__(self, num_models, num_features, hidden_size, num_classes):
        super(RiskScoreEnsembleModel, self).__init__()
        
        # backbone (MLP)
        backbone = nn.Sequential(
                                nn.Linear(num_features, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, num_classes)
                                )
        
        # models for ensemble learning
        self.models = nn.ModuleList([backbone for model in range(num_models)])
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs)


# model 3: RiskScoreModel + Ensemble Learning with PyTorch Lightning
class RiskScoreEnsembleModel_PL(pl.LightningModule):
    def __init__(self, num_models, num_features, hidden_size, num_classes, lr):
        super().__init__()
        self.num_classes = num_classes # number of classes

        # layers
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, num_classes)
        self.relu = nn.ReLU() # activation function (no learnable parameters)
        self.dropout = nn.Dropout(0.2) # dropout layer

        # hyperparameters for training
        self.lr = lr # learning rate

        # model architecture
        self.model = nn.Sequential(
            self.fc1,
            self.relu,
            self.dropout,
            self.fc2,
            self.relu,
            self.dropout,
            self.fc3
            )
        
        # multiple models for ensemble learning
        self.models = nn.ModuleList([self.model for _ in range(num_models)])
    
    def forward(self, x):
        '''
        forward function is for making predictions
        '''
        outputs = [model(x) for model in self.models] # append output from each model
        outputs = torch.stack(outputs, dim=0) # stack each output per row (dim=0) as model
        outputs = torch.mean(outputs, dim=0) # average outputs across models (dim=0)
        return outputs

    def training_step(self, batch, batch_idx):
        '''
        training_step is for training the model
        '''
        x, y = batch
        y_logits = self(x) # predictions; self(): calls forward function
        loss = F.cross_entropy(y_logits, y) # loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        '''
        validation_step is for validation
        '''
        x, y = batch
        y_logits = self(x)
        loss = F.cross_entropy(y_logits, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        '''
        test_step is for testing
        '''
        x, y = batch
        y_logits = self(x)
        loss = F.cross_entropy(y_logits, y)
        self.log('test_loss', loss)
        accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(x.device)
        acc = accuracy(y_logits, y)
        self.log('test_accuracy [%]', acc*100)

        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
