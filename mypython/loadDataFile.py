import numpy as np

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
