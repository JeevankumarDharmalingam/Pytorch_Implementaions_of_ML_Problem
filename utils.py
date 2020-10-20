import torch
import torch.nn as nn


class MOAData:
    def __init__(self,features,targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self, item):
        return {
            "X": torch.tensor(self.features[item, :], dtype=torch.float),
            "Y": torch.tensor(self.targets[item, :], dtype=torch.float)

        }

class Engine:
    def __init__(self,model,optimizer,device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def loss_fn(self,output,targets):
        return nn.BCEWithLogitsLoss()(output, targets)

    def train(self,dataloader):
        self.model.train()
        final_loss = 0
        for data in dataloader:
            self.optimizer.zero_grad()
            inputs = data["X"].to(self.device)
            outputs = data["Y"].to(self.device)

            targets = self.model(inputs)

            loss = self.loss_fn(outputs,targets)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()

        return final_loss/len(dataloader)

    def eval(self, dataloader):
        self.model.eval()
        final_loss = 0
        for data in dataloader:

            inputs = data["X"].to(self.device)
            outputs = data["Y"].to(self.device)

            targets = self.model(inputs)

            loss = self.loss_fn(outputs, targets)

            final_loss += loss.item()

        return final_loss / len(dataloader)


class Model(nn.Module):
    def __init__(self,nfeatures,ntargets,nlayers,hidden_size, dropout):
        super().__init__()
        layers = []
        for _ in range(nlayers):
            if len(layers) == 0:
                layers.append(nn.Linear(nfeatures,hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))

            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_size,ntargets))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)