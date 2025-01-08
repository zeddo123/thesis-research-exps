import torch

print(torch.cuda.is_available())

device = torch.cuda.current_device()

torch.cuda.device(0)

print(torch.cuda.device_count())

