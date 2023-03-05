"""
Class: torchNet
created by: arthicha srisuchinnawong
e-mail: arsri21@student.sdu.dk
data: 18 july 2022

torch-based neural network, a template for other neural modules
"""

# ------------------- import modules ---------------------

# standard modules
import colorama

colorama.init(autoreset=True)

# math-related modules
import torch  # cpu & gpu array


# plot

# ------------------- configuration variables ---------------------

# ------------------- class torchNet ---------------------

class torchNet(torch.nn.Module):

    # -------------------- class variable -----------------------
    # (all private)

    # -------------------- constructor -----------------------
    # (private)

    def __init__(self, device=None):
        super().__init__()

        # device
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    # -------------------- initialize matrix -----------------------
    # (private)

    def zeros(self, column, row, grad=False):
        if grad:
            return torch.nn.Parameter(torch.zeros((column, row)).to(self.device), requires_grad=True)
        else:
            return torch.zeros((column, row)).to(self.device)

    def identity(self, size, grad=False):
        if grad:
            return torch.nn.Parameter(torch.eye(size).to(self.device), requires_grad=True)
        else:
            return torch.eye(size).to(self.device)

    # -------------------- conversion -----------------------
    # (private)

    def torch(self, x):
        return x if torch.is_tensor(x) else torch.FloatTensor(x).to(self.device)

    def numpy(self, x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else x
