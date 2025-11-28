import numpy as np
import matplotlib.pyplot as plt

N = 1024 

inp  = np.loadtxt("frft_input_real.csv",  delimiter=",")
reco = np.loadtxt("frft_recon_real.csv", delimiter=",")


fig, axs = plt.subplots(1, 3, figsize=(15, 4))

im0 = axs[0].imshow(inp, origin='lower',  cmap="terrain")
axs[0].set_title("Input (real)")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(reco,origin='lower', cmap="terrain")
axs[1].set_title("Reconstructed (real)")
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(reco - inp,origin='lower', cmap="winter")
axs[2].set_title("Difference (reco - inp)")
plt.colorbar(im2, ax=axs[2])

plt.tight_layout()


N = 1024 


inp  = np.loadtxt("frft_input_imag.csv",  delimiter=",")
reco = np.loadtxt("frft_recon_imag.csv", delimiter=",")

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

im0 = axs[0].imshow(inp, origin='lower',  cmap="terrain")
axs[0].set_title("Input (img)")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(reco,origin='lower', cmap="terrain")
axs[1].set_title("Reconstructed (img)")
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(reco - inp,origin='lower', cmap="winter")
axs[2].set_title("Difference (reco - inp)")
plt.colorbar(im2, ax=axs[2])

plt.tight_layout()

plt.show()
