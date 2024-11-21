import torch
from Net_ves_factor import  pdeNet_Ves_factor_periodic # from file import model class

# convert MATLAB's numpy array into PyTorch tensor
input_shape = torch.from_numpy(input_shape).float()

model = pdeNet_Ves_factor_periodic(14, 2.9)
model.load_state_dict(torch.load("/Users/gokberk/Documents/GitHub/Ves2Dpy/matlab/shannets/ves_relax_DIFF_June8_625k_dt1e-5.pth", map_location="cpu"))
model.eval()
  
predicted_shape = (model(input_shape))
predicted_shape = predicted_shape.detach().numpy()
     