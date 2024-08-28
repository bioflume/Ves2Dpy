import numpy as np
import torch
torch.set_default_dtype(torch.float32)
from model_zoo.Net_ves_relax_midfat import Net_ves_midfat
from model_zoo.Net_ves_adv_fft import Net_ves_adv_fft
from model_zoo.Net_ves_merge_adv import Net_merge_advection
from model_zoo.Net_ves_factor import pdeNet_Ves_factor_periodic
from model_zoo.Net_ves_selften import Net_ves_selften
from model_zoo.Net_ves_merge_advten import Net_ves_merge_advten
from model_zoo.Net_ves_merge_nocoords_nearFourier import Net_ves_merge_nocoords_nearFourier

class RelaxNetwork:
    '''
    Input size (nv, 2, 128), 2 channels for x and y coords
    Output size (nv, 2, 128), 2 channels for delta_x and delta_y coords, N is dataset size
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
        model = pdeNet_Ves_factor_periodic(14, 2.9)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
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
        y_mean = self.out_param[2]
        y_std = self.out_param[3]
        
        Xin[:N] = (Xin[:N] - x_mean) / x_std
        Xin[N:] = (Xin[N:] - y_mean) / y_std

        XinitShape = torch.zeros((nv, 2, 128), dtype=torch.float64).to(self.device)
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

        DXout = torch.zeros((2*N, nv), dtype=torch.float64).to(self.device)
        DXout[:N] = (DXpred[:, 0, :] * out_x_std + out_x_mean).T
        DXout[N:] = (DXpred[:, 1, :] * out_y_std + out_y_mean).T
        return DXout
    
    def forward(self, Xin):
        Xin_copy = Xin.clone()
        input = self.preProcess(Xin)
        self.model.eval()
        with torch.no_grad():
            DXpred = self.model(input.float())
        DX = self.postProcess(DXpred.double())
        DX = DX / 1E-5 * self.dt
        Xpred = Xin_copy + DX
        return Xpred


class MergedAdvNetwork:
    '''
    For each fourier  mode, 
    Input size (nv, 2, 256), 1st channels for coords, 2nd channel for fourier basis
    Output size (nv, 2, 256), 2 channels for real and imag part
    '''
    def __init__(self, input_param, out_param, model_path, device):
        self.input_param = input_param # of shape (127, 4)
        self.out_param = out_param # of shape (127, 4)
        # self.model_path = model_path
        self.device = device
        self.model = self.loadModel(model_path)
        
    def loadModel(self, model_path):
        # s = 2
        # t = 128
        # rep = t - s + 1 # number of repetitions
        # # prepare the network
        # model = Net_merge_advection(12, 1.7, 20, rep=rep)
        # dicts = []
        # # models = []
        # for l in range(s, t+1):
        #     model_path = path + "/ves_fft_mode"+str(l)+".pth"
        #     dicts.append(torch.load(model_path, map_location = self.device))

        # # organize and match trained weights
        # dict_keys = dicts[-1].keys()
        # new_weights = {}
        # for key in dict_keys:
        #     key_comps = key.split('.')
        #     if key_comps[-1][0:3] =='num':
        #         continue
        #     params = []
        #     for dict in dicts:
        #         params.append(dict[key])
        #     new_weights[key] = torch.concat(tuple(params),dim=0)
        # model.load_state_dict(new_weights, strict=True)

        model = Net_merge_advection(12, 1.7, 20, rep=127)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        # model.load_state_dict(torch.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy/trained/ves_merged_adv.pth"))
        model.eval()
        model.to(self.device)
        return model
    
    def forward(self, X):
        N = X.shape[0]//2
        nv = X.shape[1]
        device = self.device
        s = 2
        t = 128
        rep = t - s + 1 # number of repetitions

        X = X.reshape(2*N, nv, 1)
        multiX = X.repeat_interleave(rep, dim=-1)

        x_mean = self.input_param[s-2:t-1][:, 0]
        x_std = self.input_param[s-2:t-1][:, 1]
        y_mean = self.out_param[s-2:t-1][:, 2]
        y_std = self.out_param[s-2:t-1][:, 3]

        coords = torch.zeros((nv, 2*rep, 128), dtype=torch.float64).to(device)
        coords[:, :rep, :] = ((multiX[:N] - x_mean) / x_std).permute(1,2,0)
        coords[:, rep:, :] = ((multiX[N:] - y_mean) / y_std).permute(1,2,0)

        # coords (N,2*rep,128) -> (N,rep,256)
        input_coords = torch.concat((coords[:,:rep], coords[:,rep:]), dim=-1)
        # prepare fourier basis
        theta = np.arange(N)/N*2*np.pi
        theta = theta.reshape(N,1)
        bases = 1/N*np.exp(1j*theta*np.arange(N).reshape(1,N))
        # specify which mode
        rr, ii = np.real(bases[:,s-1:t]), np.imag(bases[:,s-1:t])
        basis = torch.from_numpy(np.concatenate((rr,ii),axis=-1)).reshape(1,rep,256).to(device)
        # add the channel of fourier basis
        one_mode_inputs = [torch.concat((input_coords[:, [k]], basis.repeat(nv,1,1)[:,[k]]), dim=1) for k in range(rep)]
        input_net = torch.concat(tuple(one_mode_inputs), dim=1).to(device)
            
        # Predict using neural networks
        self.model.eval()
        with torch.no_grad():
            # Xpredict of size (127, nv, 2, 256)
            Xpredict = self.model(input_net.float()).reshape(-1,rep,2,256).transpose(0,1)
        
        for imode in range(s, t+1): 
            real_mean = self.out_param[imode - 2][0]
            real_std = self.out_param[imode - 2][1]
            imag_mean = self.out_param[imode - 2][2]
            imag_std = self.out_param[imode - 2][3]

            # % first channel is real
            Xpredict[imode - 2][:, 0, :] = (Xpredict[imode - 2][:, 0, :] * real_std) + real_mean
            # % second channel is imaginary
            Xpredict[imode - 2][:, 1, :] = (Xpredict[imode - 2][:, 1, :] * imag_std) + imag_mean

        return Xpredict.double()

    # def TODOloadModel_grouped(self): # TO DO 
        
    #     Xpredict = torch.zeros(127, nv, 2, 256).to(device)
    #     # prepare fourier basis
    #     theta = np.arange(N)/N*2*np.pi
    #     theta = theta.reshape(N,1)
    #     bases = 1/N*np.exp(1j*theta*np.arange(N).reshape(1,N))

    #     for i in range(127//num_modes+1):
    #         # from s mode to t mode, both end included
    #         s = 2 + i*num_modes
    #         t = min(2 + (i+1)*num_modes -1, 128)
    #         print(f"from mode {s} to mode {t}")
    #         rep = t - s + 1 # number of repetitions
    #         Xstand = Xstand.reshape(2*N, nv, 1)
    #         multiX = torch.from_numpy(Xstand).float().repeat(1,1,rep)

    #         x_mean = self.advNetInputNorm[s-2:t-1][:,0]
    #         x_std = self.advNetInputNorm[s-2:t-1][:,1]
    #         y_mean = self.advNetInputNorm[s-2:t-1][:,2]
    #         y_std = self.advNetInputNorm[s-2:t-1][:,3]

    #         coords = torch.zeros((nv, 2*rep, 128)).to(device)
    #         coords[:, :rep, :] = ((multiX[:N] - x_mean) / x_std).permute(1,2,0)
    #         coords[:, rep:, :] = ((multiX[N:] - y_mean) / y_std).permute(1,2,0)

    #         # coords (N,2*rep,128) -> (N,rep,256)
    #         input_coords = torch.concat((coords[:,:rep], coords[:,rep:]), dim=-1)
    #         # specify which mode
    #         rr, ii = np.real(bases[:,s-1:t]), np.imag(bases[:,s-1:t])
    #         basis = torch.from_numpy(np.concatenate((rr,ii),axis=-1)).float().reshape(1,rep,256).to(device)
    #         # add the channel of fourier basis
    #         one_mode_inputs = [torch.concat((input_coords[:, [k]], basis.repeat(nv,1,1)[:,[k]]), dim=1) for k in range(rep)]
    #         input_net = torch.concat(tuple(one_mode_inputs), dim=1).to(device)

    #         # prepare the network
    #         model = Net_merge_advection(12, 1.7, 20, rep=rep)
    #         dicts = []
    #         models = []
    #         for l in range(s, t+1):
    #             # path = "/work/09452/alberto47/ls6/vesicle/save_models/ves_fft_models/ves_fft_mode"+str(i)+".pth"
    #             path = "/work/09452/alberto47/ls6/vesToPY/Ves2Dpy/ves_adv_trained/ves_fft_mode"+str(l)+".pth"
    #             dicts.append(torch.load(path, map_location=device))
    #             subnet = Net_ves_adv_fft(12, 1.7, 20)
    #             subnet.load_state_dict(dicts[-1])
    #             models.append(subnet.to(device))

    #         # organize and match trained weights
    #         dict_keys = dicts[-1].keys()
    #         new_weights = {}
    #         for key in dict_keys:
    #             key_comps = key.split('.')
    #             if key_comps[-1][0:3] =='num':
    #                 continue
    #             params = []
    #             for dict in dicts:
    #                 params.append(dict[key])
    #             new_weights[key] = torch.concat(tuple(params),dim=0)
    #         model.load_state_dict(new_weights, strict=True)
    #         model.eval()
    #         model.to(device)

    #         # Predict using neural networks
    #         with torch.no_grad():
    #             Xpredict[s-2:t-1] = model(input_net).reshape(-1,rep,2,256).transpose(0,1)

        

class TenSelfNetwork:
    '''
    Input size (nv, 2, 128), 2 channels for x and y coords
    Output size (nv, 1, 128), 1 channel
    '''
    def __init__(self, input_param, out_param, model_path, device):
        self.input_param = input_param # contains 4 numbers
        self.out_param = out_param # contains 2 numbers
        # self.model_path = model_path
        self.device = device
        self.model = self.loadModel(model_path)
    
    def loadModel(self, model_path):
        model = Net_ves_selften(12, 2.4, 24)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
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
        XinitShape = torch.zeros((nv, 2, 128), dtype=torch.float64).to(self.device)
        for k in range(nv):
            XinitShape[k, 0, :] = (Xin[:N, k] - x_mean) / x_std
            XinitShape[k, 1, :] = (Xin[N:, k] - y_mean) / y_std
        # XinitConv = torch.from_numpy(XinitShape).float()
        return XinitShape.float()
    
    def postProcess(self, pred):
        # Xout has shape (nv, 2, N)
        N = pred.shape[2]
        nv = pred.shape[0]
        out_mean = self.out_param[0]
        out_std = self.out_param[1]

        tenPred = torch.zeros((N, nv)).to(self.device)
        for k in range(nv):
            tenPred[:,k] = (pred[k] * out_std + out_mean)
        return tenPred
    
    def forward(self, Xin):
        input = self.preProcess(Xin)
        with torch.no_grad():
            pred = self.model(input)
        tenPredstand = self.postProcess(pred)
        return tenPredstand


class MergedTenAdvNetwork:
    '''
    For each mode,
    Input size (nv, 2, 128), 2 channels for x and y coords
    Output size (nv, 2, 128), 2 channels for real and imag 
    '''
    def __init__(self, input_param, out_param, model_path, device):
        self.input_param = input_param # size (127, 4)
        self.out_param = out_param # size (127, 4)
        # self.model_path = model_path
        self.device = device
        self.model = self.loadModel(model_path)
    
    def loadModel(self, model_path):
        # s = 2
        # t = 128
        # rep = t - s + 1 # number of repetitions
        # # prepare the network
        # model = Net_ves_merge_advten(12, 2.5, 24, rep=rep)
        # dicts = []
        # # models = []
        # for l in range(s, t+1):
        #     model_path = path + "/ves_advten_mode"+str(l)+".pth"
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
        model = Net_ves_merge_advten(12, 2.5, 24, rep=127)
        # model.load_state_dict(torch.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy/trained/ves_merged_advten.pth"))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model
    
    def preProcess(self, Xin):
        # Normalize input
        nv = Xin.shape[-1]
        input = Xin[:, None].repeat_interleave(127, dim=-1).reshape(2, 128, nv, 127)
        # use broadcasting
        input[0] = (input[0] - self.input_param[:, 0])/self.input_param[:, 1]
        input[1] = (input[1] - self.input_param[:, 2])/self.input_param[:, 3]
        
        return input.permute(2,3,0,1).reshape(nv, 2*127, -1)
    
    def forward(self, inp):
        nv = inp.shape[0]
        # input = torch.from_numpy(input).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            out =  self.model(inp.float())
        
        return out.reshape(nv, 127, 2, -1).permute(2, 3, 0, 1)
    
    def postProcess(self, out):
        # out shape : (2, 128, nv, 127)
        # use broadcasting
        output = torch.zeros_like(out, dtype=torch.float64).to(self.device) # 
        output[0] = out[0] * self.out_param[:, 1] + self.out_param[:, 0]
        output[1] = out[1] * self.out_param[:, 3] + self.out_param[:, 2]

        return output.permute(3, 2, 0, 1) # shape: (127, nv, 2, 128)
        

class MergedNearFourierNetwork:
    '''
    For each mode,
    Input size (nv, 2, 128), 2 channels for x and y coords
    Output size (nv, 12, 128), 12 channels are 
        (vx_real_layer0, vx_real_layer1, vx_real_layer2, 
        vy_real_layer0, vy_real_layer1, vy_real_layer2,
        vx_imag_layer0, vx_imag_layer1, vx_imag_layer2, 
        vy_imag_layer0, vy_imag_layer1, vy_imag_layer2) for three layers
    '''
    def __init__(self, input_param, out_param, model_path, device):
        self.input_param = input_param # size (128, 4)
        self.out_param = out_param # size (128, 2, 12)
        # self.model_path = model_path
        self.device = device
        self.model = self.loadModel(model_path)
    
    def loadModel(self, path):
        # s = 1
        # t = 128
        # rep = t - s + 1 # number of repetitions
        # # prepare the network
        # model = Net_ves_merge_nocoords_nearFourier(14, 3.2, 28, rep=rep)
        # dicts = []
        # # models = []
        # for l in range(s, t+1):
        #     model_path = f"/work/09452/alberto47/ls6/vesicle_nearF2024/trained_nocoords/Ves_nearFourier_nocoords_mode{l}.pth"
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

        model = Net_ves_merge_nocoords_nearFourier(14, 3.2, 28, rep=128)
        # model.load_state_dict(torch.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy/trained/ves_merged_nearFourier.pth"))
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model
    
    def preProcess(self, Xin):
        # Normalize input
        nv = Xin.shape[-1]
        input = Xin[:, None].repeat_interleave(128, dim=-1).reshape(2, 128, nv, 128)
        # use broadcasting
        input[0] = (input[0] - self.input_param[:, 0])/self.input_param[:, 1]
        input[1] = (input[1] - self.input_param[:, 2])/self.input_param[:, 3]
        
        return input.permute(2,3,0,1).reshape(nv, 2*128, -1).float()
    
    def forward(self, input):
        nv = input.shape[0]
        # input = torch.from_numpy(input).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            out =  self.model(input)
        
        # out = out
        return out.reshape(nv, 128, 12, -1).permute(3, 0, 1, 2)
    
    def postProcess(self, out):
        # out shape : (128, nv, num_modes=128, 12)
        # use broadcasting
        out = out * self.out_param[:, 1] + self.out_param[:, 0]

        reshaped_out =  out.permute(1, 0, 2, 3) # shape: (nv, 128, num_modes=128, 12)
        # after postprocess, output velx_real, vely_real, velx_imag, vely_imag
        return reshaped_out[..., :3], reshaped_out[..., 3:6], reshaped_out[..., 6:9], reshaped_out[..., 9:]
        
        
