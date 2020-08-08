import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import copy as copy


def poly(znew, steps, degree):
	p = np.poly1d(np.polyfit(steps, znew, degree))
	return p



data = pd.read_csv("MoNbTaWV_3D_silea.txt", delimiter='\t', skiprows=8, header=None)

y, x = np.mgrid[0: 1002, 0:51]
z = np.empty([0, 1002])
zlevel = np.empty([0, 1002])
steps = np.arange(0, 1002)
fig, (ax1, ax2) = plt.subplots(nrows=2)


for i in range(0, 51): 
	znew =  data.loc[i, :].astype(float)
	z = np.vstack((z, znew))

zlevel = copy.deepcopy(z)
mask = copy.deepcopy(z)
vert = np.arange(65,75)
mask[38, vert] = mask[20, vert]

zlevel[:, 0]=zlevel[:, 1]

for degree in range(0, 2):
	
	for i in range(0, 51): 
		if degree ==1:
			pi = poly(zlevel[i, :], steps, degree)
		else:
			pi = poly(zlevel[i, :], steps, degree)
			
		zlevel[i, :] = zlevel[i, :] - pi(steps)
		
		
	for j in range(0, 1002):
		 
		if degree ==1:
			pj = poly(zlevel[:, j], steps[0:51], degree)
		else:
			pj = poly(zlevel[:, j], steps[0:51], degree)
			
		zlevel[:, j] = zlevel[:, j]- pj(steps[0:51])



levels = MaxNLocator(nbins=1001).tick_values((z.min()-z.max())*1e-4, 0)
cmap = plt.get_cmap('ocean')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
im1 = ax1.pcolormesh(y, x*20, np.transpose(z)*1e-4, cmap=cmap, norm=norm)
fig.colorbar(im1, ax=ax1, label='\u03BCm')

levels = MaxNLocator(nbins=1001).tick_values((zlevel.min()-zlevel.max()+12000)*1e-4, 0)
cmap = plt.get_cmap('ocean')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
im2 = ax2.pcolormesh(y, x*20, np.transpose(zlevel)*1e-4, cmap=cmap, norm=norm)
fig.colorbar(im2, ax=ax2, label='\u03BCm')



ticks = [0, 200, 400, 600, 800, 1000]
ax1.set_title('Smooth sample (B): Raw data ')
ax2.set_title('Smooth sample (B): Major curvature removed ')
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax2.set_xticks(ticks)
ax2.set_yticks(ticks)

ax1.set_ylabel("y \u03BCm")
ax2.set_ylabel("y \u03BCm")
ax1.set_xlabel("x \u03BCm")
ax2.set_xlabel("x \u03BCm")

fig.tight_layout()

plt.show()

