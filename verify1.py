import torch

print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.tensor([1.0, 2.0, 3.0], device=device)

print("Device:", x.device)
print("Result:", x * 2)