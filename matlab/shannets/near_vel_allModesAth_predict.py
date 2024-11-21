import torch
from Net_ves_merge_nocoords_nearFourier_disth import Net_ves_merge_nocoords_nearFourier # from file import model class

# vesicle coordinates are normalized
# convert MATLAB's numpy array into PyTorch tensor
input_shape = torch.from_numpy(input_shape).float()

model = Net_ves_merge_nocoords_nearFourier(13, 3.0, 26, rep=128)
model.load_state_dict(torch.load("/Users/gokberk/Documents/GitHub/Ves2Dpy/matlab/shannets/ves_merged_disth_nearFourier.pth",map_location="cpu"))
model.eval()
output_list = model(input_shape)
output_list = output_list.detach().numpy()
