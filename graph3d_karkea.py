import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import copy as copy
from scipy.optimize import curve_fit
from scipy import interpolate

def poly(znew, steps, degree):
	p = np.poly1d(np.polyfit(steps, znew, degree))
	return p


data = pd.read_csv("MoNbTaWV_3D_karkea.txt", delimiter='\t', skiprows=8, header=None)

y, x = np.mgrid[0: 3002, 0:21]
z = np.empty([0, 3002])
zlevel = np.empty([0, 3002])
steps = np.arange(0, 3002)
fig, (ax1, ax2) = plt.subplots(nrows=2)

for i in range(0, 21): 
	znew =  data.loc[i, :].astype(float)
	z = np.vstack((z, znew))


zlevel = copy.deepcopy(z)
mask = copy.deepcopy(z)
vert = np.arange(2650,2940)
mask[8:11, vert] = mask[7, vert]
mask[11:13, vert] = mask[14, vert]
mask[2:3, 1150:1180] = mask[4:5, 1100:1130]


zlevel[:, 0]=zlevel[:, 1]
print(zlevel[:, 0])

for degree in range(0, 1):
	
	for i in range(0, 21): 
		pi = poly(mask[i, :], steps, degree)
		zlevel[i, :] = zlevel[i, :] - pi(steps)
		
		
	for j in range(0, 3002):
		pj = poly(mask[:, j], steps[0:21], degree)
		zlevel[:, j] = zlevel[:, j]- pj(steps[0:21])



levels = MaxNLocator(nbins=501).tick_values((z.min()-z.max())*1e-4,  0)
cmap = plt.get_cmap('ocean')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
im1 = ax1.pcolormesh(y, x*1000/20, (np.transpose(z)-z.max())*1e-4, cmap=cmap, norm=norm)
fig.colorbar(im1, ax=ax1, label='\u03BCm')

levels = MaxNLocator(nbins=501).tick_values(zlevel.min()*1e-4-zlevel.max()*1e-4, 0)
cmap = plt.get_cmap('ocean')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
im2 = ax2.pcolormesh(y, x*1000/20, (np.transpose(zlevel)-zlevel.max())*1e-4, cmap=cmap, norm=norm)
fig.colorbar(im2, ax=ax2, label='\u03BCm')



ax1.set_title('Rough sample (A): Raw data ')
ax2.set_title('Rough sample (A): Major curvature removed')

#ticksx = [0, 200, 400, 600, 800, 1000]
ticksy = [0, 200, 400, 600, 800, 1000]
#ax1.set_xticks(ticksx)
ax1.set_yticks(ticksy)
#ax2.set_xticks(ticksx)
ax2.set_yticks(ticksy)

ax1.set_ylabel("y \u03BCm")
ax2.set_ylabel("y \u03BCm")
ax1.set_xlabel("x \u03BCm")
ax2.set_xlabel("x \u03BCm")

fig.tight_layout()

plt.show()


