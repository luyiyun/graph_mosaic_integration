import torch
import torch.nn as nn

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x