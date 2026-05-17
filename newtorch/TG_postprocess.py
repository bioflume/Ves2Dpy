# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import os
# from mypython.curve import Curve
from tqdm import tqdm
import multiprocessing

# %matplotlib inline
def load_single_ves_file_py(filename):
    with open(filename, 'rb') as f:
        val = np.fromfile(f, dtype=np.float64)
    
    N = int(val[0])
    nv = int(val[1])
    Xinit = val[2:2 + 2 * N * nv]
    xinit = np.zeros((N, nv))
    yinit = np.zeros((N, nv))
    
    istart = 0
    for iv in range(nv):
        iend = istart + 2*N
        X = Xinit[istart:iend]
        xinit[:, iv] = X[:N]
        yinit[:, iv] = X[N:]
        istart = iend
    # Delete the initial entries
    val = val[2 + 2 * N * nv:]
    # Number of time steps saved in the file
    ntime = len(val) // (2 * N * nv + 1) 
    # Initialize the files to save X and Y coordinates of vesicles
    # as well as time 
    
    vesx = np.zeros((N, nv, ntime))
    vesy = np.zeros((N, nv, ntime))
    time = np.zeros(ntime)
    
    istart = 0
    for it in range(ntime):
        time[it] = val[istart]
        istart += 1
        for iv in range(nv):
            iend = istart + 2 * N
            X = val[istart:iend]
            vesx[:, iv, it] = X[:N]
            vesy[:, iv, it] = X[N:]
            istart = iend
    
    return vesx, vesy, time, N, nv, xinit, yinit


def plot_vesicle_data(fileName1, name):
    if not os.path.exists("output"+"/"+name):
        os.makedirs("output/"+name)

    # Load data
    vesx1, vesy1, time, N, nv, xinit1, yinit1 = load_single_ves_file_py(fileName1)
    # vesx2, vesy2, time, N, nv, xinit2, yinit2 = load_single_ves_file_py(fileName2)
    # np.save("TG_from_BIEM_-100.npy", np.concatenate((vesx[:,:,-100],vesy[:,:,-100]), axis=0))
    plt.figure(1, figsize=(8,8))
    plt.clf()
    
    # Concatenate the first point to the end to close the loop
    x1 = np.vstack((xinit1, xinit1[0, :]))
    y1 = np.vstack((yinit1, yinit1[0, :]))
    # x2 = np.vstack((xinit2, xinit2[0, :]))
    # y2 = np.vstack((yinit2, yinit2[0, :]))
    
    plt.plot(x1, y1, 'r', linewidth=2)
    # plt.plot(x2, y2, 'grey', linewidth=2)
    # plt.plot(x[0], y[0], 'ko', markersize=10, markerfacecolor='k')
    # plt.plot(np.mean(x), np.mean(y), 'kx', markerfacecolor='k')
    plt.axis('scaled')
    plt.xlim([0, 2.5])
    plt.ylim([0, 2.5])
    
    # plt.show()
    plt.pause(0.01)
    plt.close() 
    # plt.savefig(f"output/{name}/0.png", dpi=50, transparent=False)

    # oc = Curve()
    # np.save("problematic_laststeps_nv128.npy", np.concatenate((vesx[:, :, -10:], vesy[:, :, -10:]), axis=0))
    # from sys import exit
    # exit(0)

    for it in tqdm(range(len(time))): # len(time) # 25000
        # img = plt.imread("/work/09452/alberto47/vista/Ves2Dpy/Speed-200_l-0.5_name-combination_file1.png")
        
        fig, ax = plt.subplots()

        # Show image
        # ax.imshow(img, extent=[-4, 10, 0, 14], alpha=1)

        # plt.figure(figsize=(10,10), dpi=50)
        # plt.clf()
        # plt.imshow(img, extent=[-12, 4, -12, 4], alpha=0.5)

        # Concatenate the first point to the end to close the loop
        # print(np.hstack((vesx[:,:,-1], vesy[:,:,-1])).shape)
        # np.save("single_parabolic_nofilter.npy", np.hstack((vesx[:,:,-1], vesy[:,:,-1])))
        # return
        x1 = np.vstack((vesx1[:, :, it], vesx1[0, :, it]))
        y1 = np.vstack((vesy1[:, :, it], vesy1[0, :, it]))

        # x2 = np.vstack((vesx2[:, :, it], vesx2[0, :, it]))
        # y2 = np.vstack((vesy2[:, :, it], vesy2[0, :, it]))

        # N = 128
        # max_layer_dist = np.sqrt(1/N)
        # _, tang, _ = oc.diffProp(np.concatenate((vesx[:, :, it],vesy[:, :, it])))
        # # Normal
        # nx = tang[N:2*N, :]
        # ny = -tang[:N, :]
        # Find the outermost layers of every vesicle
        # Xlarge = np.zeros((2 * N, nv))
        # for k in range(nv):
        #     Xlarge[:, k] = np.concatenate([vesx[:, k, it] + nx[:, k] * max_layer_dist, 
        #                                 vesy[:, k, it] + ny[:, k] * max_layer_dist])
            
        ax.plot(x1, y1, 'red', linewidth=2)
        # plt.plot(x2, y2, 'grey', linewidth=2, alpha=0.8)
        # plt.plot(x[:,0], y[:,0], 'r', linewidth=2)
        # plt.plot(x[:,1], y[:,1], 'blue', linewidth=2)
        # # plt.plot(Xlarge[:128], Xlarge[128:], 'b-', linewidth=1)
        # plt.plot(x[0], y[0], 'ko', markersize=6, markerfacecolor='k')
        # # plt.plot(np.mean(x), np.mean(y), 'kx', markerfacecolor='k')
        # # plt.text(x[3,0], y[3,0], "3", ha='center')
        # plt.text(x[7,0], y[7,0], "7", ha='center')
        # plt.text(x[32,0], y[32,0], "32", ha='center')
        # plt.text(x[64,0], y[64,0], "64", ha='center')
        # plt.text(x[96,0], y[96,0], "96", ha='center')
        # plt.text(x[110,0], y[110,0], "110", ha='center')

        # plt.text(x[4,1], y[4,1], "4", ha='center')
        # plt.text(x[8,1], y[8,1], "8", ha='center')
        # plt.text(x[30,1], y[30,1], "30", ha='center')
        # plt.text(x[64,1], y[64,1], "64", ha='center')
        # plt.text(x[90,1], y[90,1], "90", ha='center')
        # plt.text(x[110,1], y[110,1], "110", ha='center')

        plt.axis('scaled')
        # plt.axis('off')
        # plt.xlim([-0.6 * (np.max(x) - np.min(x)) + np.max(x), np.max(x)])
        plt.xlim([0, 2.5])
        plt.ylim([0, 2.5])
        plt.title(f"t = {it+1}"+" "+name)
        
        # plt.show()
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # plt.savefig(f"output/{name}/{it+1}.png", dpi=50, transparent=False)
        plt.savefig(f"output/{name}/{it+1}.png")
        
        plt.pause(0.0001)
        plt.close() 
    
    
if __name__ == "__main__":
    # jobid = str(239946)
    # plot_vesicle_data("output/job" + jobid + ".bin", name = "job" + jobid)
    # plot_vesicle_data('./output_N128/does_near_help_without.bin', name = "does_near_help_without")
    # plot_vesicle_data('./output_N128/lsls.bin', name = "linshi_N128")
    # plot_vesicle_data('./output/parabolic_shape8_nv2000_auglag.bin', name = "sep1_parabolic_moving_window")
    # plot_vesicle_data('./output/parabolic_shape8_nv2000_auglag.bin', name = "parabolic_25fall")

    # for fileIdx in range(5, 12):
    #     plot_vesicle_data(f'./output_BIEM/gnn_training_single{fileIdx}.bin', name = f"GNN_training_single{fileIdx}")
    # plot_vesicle_data('./output_BIEM/gnn_training_single1.bin', name = "GNN_training_single1")
    # plot_vesicle_data(f'./output_BIEM/gnn_training_round_{10}.bin', name = "GNN_training_round_10")
    # plot_vesicle_data(f'./output/gnn_compare_vesnet_single_AUG10.bin', name = "GNN_vesnet_single_AUG10")
    plot_vesicle_data(f'../mytorch/output_BIEM/GNN_training_wv/gnn_training_normal_size_sampleW_single_wv_ext-96.35542452037956_farFieldSpeed-0.0_name-sampleW_rot-305.5341809743935_file2360.bin', name = "GNN_normal_size_sampleW_2360")

    # plot_vesicle_data('./output/TG_nv32_N32_25fall.bin', './output_BIEM/TG_nv32_VF25.bin', name = "TG_nv32_N32_25fall")
    # plot_vesicle_data('./output_BIEM/ls_N128_noNear.bin', name = "shan_BIEM_N128_without")
    # plot_vesicle_data('./output_BIEM/BIEM_N128_TG_VF25_35ksteps.bin', name="linshi_biem_N128")
    # plot_vesicle_data("output/ls.bin", name = "ls")
    # plot_vesicle_data("./output/TG_nv32_N32_25fall.bin", name = "TG_nv32_VF25_25fall")
    # plot_vesicle_data("./output_BIEM/TG_nv32_VF25.bin", name = "TG_BIEM_nv32_VF25_25fall")
    # plot_vesicle_data("output/VF25_TG128Ves.bin", name = "TG_N32_dilute_nv128")
    # plot_vesicle_data("output/VF12_TG_240Ves.bin", name = "TG_N32_diluteVF12_nv240")
    # plot_vesicle_data("output/VF12_TG_2220Ves.bin", name = "TG_N32_diluteVF12_nv2220")
    # plot_vesicle_data("output/linshi.bin", name = "TG_N32_dilute_nv32_VF25_3layers_rbf_upsample2")
    # plot_vesicle_data("output/up2.bin", name = "TG_N32_dilute_nv32_VF25_5layers_rbf_upsample2")
    # plot_vesicle_data("output/lsls.bin", name = "linshi")

    # %%
