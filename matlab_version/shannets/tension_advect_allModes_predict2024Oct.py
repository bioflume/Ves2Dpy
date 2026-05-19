import torch
from Net_ves_merge_advten_2024Oct import Net_ves_merge_advten # from file import model class

# vesicle coordinates are normalized
# convert MATLAB's numpy array into PyTorch tensor
input_shape = torch.from_numpy(input_shape).float()

model = Net_ves_merge_advten(12, 2.5, 24, rep=127)
model.load_state_dict(torch.load("/Users/gokberk/Documents/GitHub/Ves2Dpy/matlab/shannets/2024Oct_ves_merged_advten.pth",map_location="cpu"))
model.eval()
output_list = model(input_shape)
output_list = output_list.detach().numpy()
