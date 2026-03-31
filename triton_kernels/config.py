import torch

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
