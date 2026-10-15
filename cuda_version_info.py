import sys
import torch
import torchvision
import torchaudio

print("python version:", sys.version)
print("python version info:", sys.version_info)
print("torch version:", torch.__version__)
print("cuda version (torch):", torch.version.cuda)
print("torchvision version:", torchvision.__version__)
print("torchaudio version:", torchaudio.__version__)
print("cuda available:", torch.cuda.is_available())
