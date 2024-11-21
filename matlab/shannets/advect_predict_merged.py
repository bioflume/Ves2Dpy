import torch
from Net_ves_merge_adv import Net_merge_advection # from file import model class

# vesicle coordinates are normalized
# convert MATLAB's numpy array into PyTorch tensor
input_shape = torch.from_numpy(input_shape).float()
model = Net_merge_advection(12, 1.7, 20, rep=127)
model.load_state_dict(torch.load("/Users/gokberk/Documents/GitHub/Ves2Dpy/matlab/shannets/ves_merged_adv.pth",map_location="cpu"))
model.eval()
output_list = model(input_shape)
output_list = output_list.detach().numpy()

     
  