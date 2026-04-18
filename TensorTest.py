import torch
import numpy as np

tensor = torch.zeros([3,3,3])
index_map = torch.tensor([
    [[1,1], [1,1]], 
    [[2,2], [2,2]]  # Added an extra [3] so the length is 2
])
dimension = 1

print(f"Before Scatter:\n\n{tensor}\n")

tensor.scatter_(1, index_map, value=1)

print(f"After Scatter.\nDimension = {dimension}:\n\n{tensor}\n\n")