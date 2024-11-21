import torch
from Net_ves_selftenOct24 import  Net_ves_selften # from file import model class

# convert MATLAB's numpy array into PyTorch tensor
input_shape = torch.from_numpy(input_shape).float()

model = Net_ves_selften(12, 2.4, 26)
model.load_state_dict(torch.load("/Users/gokberk/Documents/GitHub/Ves2Dpy/matlab/shannets/ves_selften_2024Oct.pth", map_location="cpu"))
model.eval()
  
predicted_shape = (model(input_shape))
predicted_shape = predicted_shape.detach().numpy()
     