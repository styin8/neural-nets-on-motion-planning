import torch.nn as nn
import torch
import math


class CFN(nn.Module):
    def __init__(self, opt):
        super(CFN, self).__init__()
        # dim
        self.hidden = opt.hidden
        self.input = opt.dof + 2 * opt.L * opt.dof
        self.layer4 = opt.hidden + opt.dof + 2 * opt.L
        self.output = opt.voxel
        # layers
        self.fc1 = nn.Linear(self.input, self.hidden)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(opt.dropout)

        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(opt.dropout)

        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(opt.dropout)

        self.fc4 = nn.Linear(self.layer4, self.hidden)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(opt.dropout)

        self.fc5 = nn.Linear(self.hidden, self.hidden)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(opt.dropout)

        self.last = nn.Linear(self.hidden, self.output)
        if opt.init_weight:
            self._initialize_weights()

    def sin_cos_kernel(self, q: torch.Tensor, L: int):
        """
        ker(q) = [sin(2^(0)πq),cos(2^(0)πq),...,sin(2^(L−1)πq),cos(2^(L−1)πq)]
        """
        result = []
        for i in range(L):
            result.append(torch.sin(math.pow(2, i) * math.pi * q))
            result.append(torch.cos(math.pow(2, i) * math.pi * q))
        return torch.cat(result, dim=1)

    def forward(self, q: torch.Tensor):
        ker_q = self.sin_cos_kernel(q, self.L)
        cat_q_kernel = torch.cat((ker_q, q), dim=1)

        x1 = self.fc1(cat_q_kernel)
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)

        x2 = self.fc2(x1)
        x2 = self.relu2(x2)
        x2 = self.dropout2(x2)

        x3 = self.fc3(x2)
        x3 = self.relu3(x3)
        x3 = self.dropout3(x3)

        before_layer4 = torch.cat((cat_q_kernel, x3), dim=1)
        x4 = self.fc4(before_layer4)
        x4 = self.relu4(x4)
        x4 = self.dropout4(x4)

        x5 = self.fc5(x4)
        x5 = self.relu5(x5)
        x5 = self.dropout5(x5)

        out = self.last(x5)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def compute_loss(self, pred, gt):
        loss = nn.L1Loss(pred, gt)
        return loss
    
    def net_train(self, dataset, opt):
        result = {
            "loss": 0,
            "acc": 0
        }
        device = torch.device(f"cuda:{opt.gpu_ids}" if torch.cuda.is_available() else "cpu")
        self.train()
        for x, y in dataset:
            xc = x.to(device)
            yc = y.to(device)
            pred = self.forward(xc)
            loss = self.compute_loss(pred, yc)