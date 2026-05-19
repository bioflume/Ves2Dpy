import numpy as np
import matplotlib.pyplot as plt

# --- Load data ---
filename = "shear.bin"
vesx, vesy, time, N, nv, xinit, yinit = load_ves2d_file(filename)

ntime = time.size

# Global bounds (same as MATLAB vesx(:), vesy(:))
xmin, xmax = vesx.min(), vesx.max()
ymin, ymax = vesy.min(), vesy.max()

# --- Time loop ---
for it in range(ntime):
    plt.figure(1)
    plt.clf()

    # Close the vesicle curves by appending the first point
    x = np.vstack([vesx[:, :, it], vesx[0, :, it]])
    y = np.vstack([vesy[:, :, it], vesy[0, :, it]])

    # Plot outline
    plt.plot(x, y, 'r', linewidth=2)

    # Filled vesicles
    plt.fill(x, y, color='r', edgecolor='r')

    # Axis settings
    plt.axis('equal')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.title(f"{it + 1}")  # MATLAB is 1-based

    plt.box(True)

    plt.pause(0.1)

plt.show()