import numpy as np
import torch
torch.set_default_dtype(torch.float32)
from model_zoo_N32.Ves_relax_downsample_zerolevel import Ves_relax_downsample_zerolevel
# from model_zoo_N32.pdeNet_Ves_fat_factor_modein5_onelevel import pdeNet_Ves_fat_factor_modein5_onelevel
from model_zoo_N32.Net_ves_merge_adv import Net_merge_advection
from model_zoo_N32.Ves_selften_downsample_zerolevel import pdeNet_Ves_fat_factor_modein6_zerolevel
from model_zoo_N32.Net_ves_merge_advten import Net_ves_merge_advten
from model_zoo_N32.Net_ves_merge_nocoords_nearFourier import Net_ves_merge_nocoords_nearFourier
from model_zoo_N32.Net_ves_merge_nocoords_innerNearFourier import Net_ves_merge_nocoords_innerNearFourier

class RelaxNetwork:
    '''
    Input size (nv, 2, 32), 2 channels for x and y coords
    Output size (nv, 2, 32), 2 channels for delta_x and delta_y coords, N is dataset size
    Note that the network predicts differences.
    '''
    def __init__(self, dt, input_param, out_param, model_path, device):
        self.dt = dt
        self.input_param = input_param # contains 4 numbers
        self.out_param = out_param # contains 4 numbers
        # self.model_path = model_path
        self.device = device
        self.model = self.loadModel(model_path)
    
    def loadModel(self, model_path):
        model = Ves_relax_downsample_zerolevel(12, 2.5)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        return model
    
    def preProcess(self, Xin):
        # Xin has shape (2N, nv)
        # XinitConv has shape (nv, 2, N)
        N = Xin.shape[0]//2
        nv = Xin.shape[1]

        x_mean = self.input_param[0]
        x_std = self.input_param[1]
        y_mean = self.input_param[2]
        y_std = self.input_param[3]
        
        Xin[:N] = (Xin[:N] - x_mean) / x_std
        Xin[N:] = (Xin[N:] - y_mean) / y_std

        XinitShape = torch.zeros((nv, 2, 32), dtype=torch.float32).to(self.device)
        XinitShape[:, 0, :] = Xin[:N].T
        XinitShape[:, 1, :] = Xin[N:].T
        # XinitConv = torch.from_numpy(XinitShape).float()
        return XinitShape
    
    def postProcess(self, DXpred):
        # Xout has shape (nv, 2, N)
        N = DXpred.shape[2]
        nv = DXpred.shape[0]
        out_x_mean = self.out_param[0]
        out_x_std = self.out_param[1]
        out_y_mean = self.out_param[2]
        out_y_std = self.out_param[3]

        DXout = torch.zeros((2*N, nv), dtype=torch.float32).to(self.device)
        DXout[:N] = (DXpred[:, 0, :] * out_x_std + out_x_mean).T
        DXout[N:] = (DXpred[:, 1, :] * out_y_std + out_y_mean).T
        return DXout
    
    def forward(self, Xin):
        Xin_copy = Xin.clone()
        input = self.preProcess(Xin)
        self.model.eval()
        with torch.no_grad():
            DXpred = self.model(input.float())
        DX = self.postProcess(DXpred)
        DX = DX / 1E-5 * self.dt
        Xpred = Xin_copy + DX
        return Xpred


class MergedAdvNetwork:
    '''
    For each fourier  mode, 
    Input size (nv, 2, 64), 1st channels for coords, 2nd channel for fourier basis
    Output size (nv, 2, 64), 2 channels for real and imag part
    '''
    def __init__(self, input_param, out_param, model_path, device):
        self.input_param = input_param # of shape (31, 4)
        self.out_param = out_param # of shape (31, 4)
        # self.model_path = model_path
        self.device = device
        self.model = self.loadModel(model_path)
        
    def loadModel(self, model_path):
        # s = 2
        # t = 32
        # rep = t - s + 1 # number of repetitions
        # # prepare the network
        # model = Net_merge_advection(12, 2.0, 26, rep=rep)
        # dicts = []
        # # models = []
        # for l in range(s, t+1):
        #     model_path = f"../trained/adv_fft_ds32/ves_adv_downsample_fft_2024Oct_mode{l}.pth"
        #     dicts.append(torch.load(model_path, map_location = self.device))

        # # organize and match trained weights
        # dict_keys = dicts[-1].keys()
        # new_weights = {}
        # for key in dict_keys:
        #     key_comps = key.split('.')
        #     if key_comps[-1][0:3] =='num':
        #         continue
        #     params = []
        #     for i, dict in enumerate(dicts):
        #         params.append(dict[key])
        #     new_weights[key] = torch.concat(tuple(params),dim=0)
        # model.load_state_dict(new_weights, strict=True)
        # torch.save(model.state_dict(), "../trained/adv_fft_ds32/2024Oct_ves_merged_adv.pth")

        model = Net_merge_advection(12, 2.0, 26, rep=31)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.eval()
        model.to(self.device)
        return model
    
    def forward(self, X):
        N = X.shape[0]//2
        nv = X.shape[1]
        device = self.device
        s = 2
        t = 32
        rep = t - s + 1 # number of repetitions

        multiX = X.unsqueeze(-1).repeat_interleave(rep, dim=-1)

        x_mean = self.input_param[s-2:t-1][:, 0]
        x_std = self.input_param[s-2:t-1][:, 1]
        y_mean = self.input_param[s-2:t-1][:, 2]
        y_std = self.input_param[s-2:t-1][:, 3]
        
        coords = torch.zeros((nv, 2*rep, 32), dtype=torch.float32).to(device)
        coords[:, :rep, :] = ((multiX[:N] - x_mean) / x_std).permute(1,2,0)
        coords[:, rep:, :] = ((multiX[N:] - y_mean) / y_std).permute(1,2,0)

        # coords (N,2*rep,32) -> input_coords (nv,rep,256)
        input_coords = torch.concat((coords[:,:rep], coords[:,rep:]), dim=-1)
        # prepare fourier basis
        theta = np.arange(N)/N*2*np.pi
        theta = theta.reshape(N,1)
        bases = 1/N*np.exp(1j*theta*np.arange(N).reshape(1,N))
        # specify which mode
        rr, ii = np.real(bases[:,s-1:t]), np.imag(bases[:,s-1:t])
        basis = torch.from_numpy(np.concatenate((rr,ii),axis=0)).T.reshape(1,rep,64).to(device)
        # add the channel of fourier basis
        # one_mode_inputs = [torch.concat((input_coords[:, [k]], basis.repeat(nv,1,1)[:,[k]]), dim=1) for k in range(rep)]
        # input_net = torch.concat(tuple(one_mode_inputs), dim=1).to(device) # input_net (nv, 254, 256)
        
        stacked = torch.stack((input_coords, basis.repeat(nv,1,1)), dim=2) # (nv, rep, 2, 2*N)
        interleaved = stacked.reshape(nv, 2*rep, 2*N).to(device)
        
        # if not torch.allclose(input_net, interleaved):
        #     raise "batch err"

        # Predict using neural networks
        self.model.eval()
        with torch.no_grad():
            # Xpredict of size (31, nv, 2, 64)
            Xpredict = self.model(interleaved.float()).reshape(-1,rep,2,64).transpose(0,1)
        
        xpred_ = torch.zeros_like(Xpredict)
        xpred_[:,:,0] = Xpredict[:,:,0] * self.out_param[:,1,None,None] + self.out_param[:,0,None,None]
        xpred_[:,:,1] = Xpredict[:,:,1] * self.out_param[:,3,None,None] + self.out_param[:,2,None,None]

        # for imode in range(s, t+1): 
        #     real_mean = self.out_param[imode - 2][0]
        #     real_std = self.out_param[imode - 2][1]
        #     imag_mean = self.out_param[imode - 2][2]
        #     imag_std = self.out_param[imode - 2][3]

        #     # % first channel is real
        #     Xpredict[imode - 2][:, 0, :] = (Xpredict[imode - 2][:, 0, :] * real_std) + real_mean
        #     # % second channel is imaginary
        #     Xpredict[imode - 2][:, 1, :] = (Xpredict[imode - 2][:, 1, :] * imag_std) + imag_mean
        
        # Xpredict[63] = torch.zeros(nv, 2, 256)

        # if not torch.allclose(xpred_, Xpredict):
        #     raise "batch err"

        return xpred_


class TenSelfNetwork:
    '''
    Input size (nv, 2, 32), 2 channels for x and y coords
    Output size (nv, 1, 32), 1 channel
    '''
    def __init__(self, input_param, out_param, model_path, device):
        self.input_param = input_param # contains 4 numbers
        self.out_param = out_param # contains 2 numbers
        # self.model_path = model_path
        self.device = device
        self.model = self.loadModel(model_path)
    
    def loadModel(self, model_path):
        model = pdeNet_Ves_fat_factor_modein6_zerolevel(12, 1.5, 20)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        return model
    
    def preProcess(self, Xin):
        # Xin has shape (2N, nv)
        # XinitConv has shape (nv, 2, N)
        N = Xin.shape[0]//2
        nv = Xin.shape[1]

        x_mean = self.input_param[0]
        x_std = self.input_param[1]
        y_mean = self.input_param[2]
        y_std = self.input_param[3]
        
        # Adjust the input shape for the network
        # XinitShape = torch.zeros((nv, 2, 32), dtype=torch.float32).to(self.device)
        # for k in range(nv):
        #     XinitShape[k, 0, :] = (Xin[:N, k] - x_mean) / x_std
        #     XinitShape[k, 1, :] = (Xin[N:, k] - y_mean) / y_std
        
        XinitShape_ = torch.zeros((nv, 2, 32), dtype=torch.float32).to(self.device)
        XinitShape_[:, 0, :] = (Xin[:N].T - x_mean) / x_std
        XinitShape_[:, 1, :] = (Xin[N:].T - y_mean) / y_std
        
        # if not torch.allclose(XinitShape, XinitShape_):
        #     raise "batch err"
        return XinitShape_.float()
    
    def postProcess(self, pred):
        # pred has shape (nv, 1, N)
        # N = pred.shape[2]
        # nv = pred.shape[0]
        out_mean = self.out_param[0]
        out_std = self.out_param[1]
        pred = pred.squeeze() # (nv, N)

        # tenPred = torch.zeros((N, nv)).to(self.device)
        # for k in range(nv):
        #     tenPred[:,k] = (pred[k] * out_std + out_mean)

        tenPred = (pred.T * out_std + out_mean)

        # if not torch.allclose(tenPred, tenPred_):
        #     raise "batch err"
        return tenPred
    
    def forward(self, Xin):
        input = self.preProcess(Xin.to(self.device))
        with torch.no_grad():
            pred = self.model(input)
        tenPredstand = self.postProcess(pred)
        return tenPredstand

class MergedTenAdvNetwork:
    '''
    For each mode,
    Input size (nv, 2, 32), 2 channels for x and y coords
    Output size (nv, 2, 32), 2 channels for real and imag 
    '''
    def __init__(self, input_param, out_param, model_path, device):
        self.input_param = input_param # size (31, 4)
        self.out_param = out_param # size (31, 4)
        # self.model_path = model_path
        self.device = device
        self.model = self.loadModel(model_path)
    
    def loadModel(self, model_path):
        # s = 2
        # t = 32
        # rep = t - s + 1 # number of repetitions
        # # prepare the network
        # model = Net_ves_merge_advten(14, 2.4, 30, rep=rep)
        # dicts = []
        # # models = []
        # for l in range(s, t+1):
        #     model_path = f"../trained/advten_downsample32/2024Nov_downsample32_ves_advten_mode{l}.pth"
        #     dicts.append(torch.load(model_path, map_location = self.device))

        # # organize and match trained weights
        # dict_keys = dicts[-1].keys()
        # new_weights = {}
        # for key in dict_keys:
        #     key_comps = key.split('.')
        #     if key_comps[-1][0:3] =='num': # skip key "num_batches_tracked" in bn 
        #         continue
        #     params = []
        #     for dict in dicts:
        #         params.append(dict[key])
        #     new_weights[key] = torch.concat(tuple(params),dim=0)
        # model.load_state_dict(new_weights, strict=True)
        # torch.save(model.state_dict(),"2024Oct_merged_advten.pth")

        model = Net_ves_merge_advten(14, 2.4, 30, rep=31)
        # model.load_state_dict(torch.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy/trained/ves_merged_advten.pth"))
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.eval()
        model.to(self.device)
        return model
    
    def preProcess(self, Xin):
        # Normalize input
        nv = Xin.shape[-1]
        input = Xin[:, None].repeat_interleave(31, dim=-1).reshape(2, 32, nv, 31).to(self.device)
        # use broadcasting
        input[0] = (input[0] - self.input_param[:, 0])/self.input_param[:, 1]
        input[1] = (input[1] - self.input_param[:, 2])/self.input_param[:, 3]
        
        return input.permute(2,3,0,1).reshape(nv, 2*31, -1)
    
    def forward(self, inp):
        nv = inp.shape[0]
        # input = torch.from_numpy(input).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            out =  self.model(inp.float())
        
        return out.reshape(nv, 31, 2, -1).permute(2, 3, 0, 1)
    
    def postProcess(self, out):
        # out shape : (2, 32, nv, 31)
        # use broadcasting
        output = torch.zeros_like(out, dtype=torch.float32).to(self.device) # 
        output[0] = out[0] * self.out_param[:, 1] + self.out_param[:, 0]
        output[1] = out[1] * self.out_param[:, 3] + self.out_param[:, 2]

        return output.permute(3, 2, 0, 1) # shape: (31, nv, 2, 32)
        

class MergedNearFourierNetwork:
    '''
    For each mode,
    Input size (nv, 2, 32), 2 channels for x and y coords
    Output size (nv, 12, 32), 12 channels are 
        (vx_real_layer0, vx_real_layer1, vx_real_layer2, 
        vy_real_layer0, vy_real_layer1, vy_real_layer2,
        vx_imag_layer0, vx_imag_layer1, vx_imag_layer2, 
        vy_imag_layer0, vy_imag_layer1, vy_imag_layer2) for three layers
    '''
    def __init__(self, input_param, out_param, model_path, device):
        self.input_param = input_param # size (32, 4)
        self.out_param = out_param # size (32, 2, 12)
        # self.model_path = model_path
        self.device = device
        self.model = self.loadModel(model_path)
    
    def loadModel(self, path):
        # s = 1
        # t = 32
        # rep = t - s + 1 # number of repetitions
        # # prepare the network
        # model = Net_ves_merge_nocoords_nearFourier(12, 1.6, 18, rep=rep)
        # dicts = []
        # # models = []
        # for l in range(s, t+1):
        #     model_path = f"../trained/near_trained/Ves_downsample_nearFourier_nocoords_mode{l}.pth"
        #     dicts.append(torch.load(model_path, map_location = self.device))

        # # organize and match trained weights
        # dict_keys = dicts[-1].keys()
        # new_weights = {}
        # for key in dict_keys:
        #     key_comps = key.split('.')
        #     if key_comps[-1][0:3] =='num': # skip key "num_batches_tracked" in bn 
        #         continue
        #     params = []
        #     for dict in dicts:
        #         params.append(dict[key])
        #     new_weights[key] = torch.concat(tuple(params),dim=0)
        # model.load_state_dict(new_weights, strict=True)
        # torch.save(model.state_dict(), "../trained/near_trained/ves_merged_disth_nearFourier.pth")
        
        
        model = Net_ves_merge_nocoords_nearFourier(12, 1.6, 18, rep=32)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.eval()
        model.to(self.device)
        return model
    
    def preProcess(self, Xin):
        # Normalize input
        nv = Xin.shape[-1]
        input = Xin[:, None].repeat_interleave(32, dim=-1).reshape(2, 32, nv, 32).to(self.device)
        
        # self.ip = self.input_param.T.reshape(2,2,1,1,-1)
        # input_ = (input - self.ip[:,0]) / self.ip[:,1]
        
        # use broadcasting
        input[0] = (input[0] - self.input_param[:, 0])/self.input_param[:, 1]
        input[1] = (input[1] - self.input_param[:, 2])/self.input_param[:, 3]
        
        # if not torch.allclose(input, input_):
        #     raise "batch err"
        return input.permute(2,3,0,1).reshape(nv, 2*32, -1).float()
    
    def forward(self, input):
        nv = input.shape[0]
        # input = torch.from_numpy(input).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            out =  self.model(input)
        
        return out.reshape(nv, 32, 12, -1).permute(3, 0, 1, 2)
    
    def postProcess(self, out):
        # out shape : (32, nv, num_modes=32, 12)
        # use broadcasting
        out = out * self.out_param[:, 1] + self.out_param[:, 0]

        reshaped_out =  out.permute(1, 0, 2, 3) # shape: (nv, 32, num_modes=32, 12)
        # after postprocess, output velx_real, vely_real, velx_imag, vely_imag
        return reshaped_out[..., :3], reshaped_out[..., 3:6], reshaped_out[..., 6:9], reshaped_out[..., 9:]
        
        

class MergedInnerNearFourierNetwork:
    '''
    For each mode,
    Input size (nv, 2, 32), 2 channels for x and y coords
    Output size (nv, 8, 32), 8 channels are 
        (vx_real_layer0, vx_real_layer1, 
        vy_real_layer0, vy_real_layer1,
        vx_imag_layer0, vx_imag_layer1,
        vy_imag_layer0, vy_imag_layer1) for two layers
    '''
    def __init__(self, input_param, out_param, model_path, device):
        self.input_param = input_param # size (32, 4)
        self.out_param = out_param # size (32, 2, 8)
        # self.model_path = model_path
        self.device = device
        self.model = self.loadModel(model_path)
    
    def loadModel(self, path):
        # s = 2
        # t = 32
        # rep = t - s + 1 # number of repetitions
        # # prepare the network
        # model = Net_ves_merge_nocoords_innerNearFourier(12, 3.2, 30, rep=rep)
        # dicts = []
        
        # for l in range(s, t+1):
        #     model_path = f"/work/09452/alberto47/ls6/vesicle_nearF2024/trained_disth_nocoords/inner_downsample32/Ves_downsample_inner_nearFourier_nocoords_mode{l}.pth"
        #     dicts.append(torch.load(model_path, map_location = self.device))

        # # organize and match trained weights
        # dict_keys = dicts[-1].keys()
        # new_weights = {}
        # for key in dict_keys:
        #     key_comps = key.split('.')
        #     if key_comps[-1][0:3] =='num': # skip key "num_batches_tracked" in bn 
        #         continue
        #     params = []
        #     for dict in dicts:
        #         params.append(dict[key])
        #     new_weights[key] = torch.concat(tuple(params),dim=0)
        # model.load_state_dict(new_weights, strict=True)
        # torch.save(model.state_dict(), "../trained/2025ves_merged_disth_innerNearFourier.pth")
        
        
        model = Net_ves_merge_nocoords_innerNearFourier(12, 3.2, 30, rep=31)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.eval()
        model.to(self.device)
        return model
    
    def preProcess(self, Xin):
        # Normalize input
        nv = Xin.shape[-1]
        input = Xin[:, None].repeat_interleave(31, dim=-1).reshape(2, 32, nv, 31).to(self.device)
        
        # self.ip = self.input_param.T.reshape(2,2,1,1,-1)
        # input_ = (input - self.ip[:,0]) / self.ip[:,1]
        
        # use broadcasting
        input[0] = (input[0] - self.input_param[:, 0])/self.input_param[:, 1]
        input[1] = (input[1] - self.input_param[:, 2])/self.input_param[:, 3]
        
        # if not torch.allclose(input, input_):
        #     raise "batch err"
        return input.permute(2,3,0,1).reshape(nv, 2*31, -1).float()
    
    def forward(self, input):
        nv = input.shape[0]
        # input = torch.from_numpy(input).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            out =  self.model(input)
        
        return out.reshape(nv, 31, 8, -1).permute(3, 0, 1, 2)
    
    def postProcess(self, out):
        # out shape : (32, nv, num_modes=31, 8)
        # use broadcasting
        out = out * self.out_param[:, 1] + self.out_param[:, 0]

        reshaped_out =  out.permute(1, 0, 2, 3) # shape: (nv, 32, num_modes=31, 8)
        # after postprocess, output velx_real, vely_real, velx_imag, vely_imag
        # return reshaped_out[..., :2], reshaped_out[..., 2:4], reshaped_out[..., 4:6], reshaped_out[..., 6:]
        return reshaped_out[..., [1,0]], reshaped_out[..., [3,2]], reshaped_out[..., [5,4]], reshaped_out[..., [7,6]]
        
        
