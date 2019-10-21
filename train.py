import numpy as np 
import torch

W = torch.rand(3, 5)

f = W * W.t()

if __name__ == "__main__":
	print("testing")