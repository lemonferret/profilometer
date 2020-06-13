import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy as copy

""" 
Profilometer 2D data-analysis: 
		-Tilt correction by linear fit
		-Surface curvature correction by multiple polynomial fitting of rising degree
		-Surface parameters: mean, rms (*?) 
"""

def poly(znew, steps, degree):											#polynom fitting
	p = np.poly1d(np.polyfit(steps, znew, degree))
	return p

def data_read(name):													#read data into numpy arrays
	data = pd.read_csv(name, delimiter='\t', skiprows=5, header=None)
	y = np.asarray(copy.deepcopy(data[0]))
	z = np.asarray(copy.deepcopy(data[1]))
	return y, z
	
def make_plots(y, zuntilt, zlevel, avg, rms, title):					#plots and specifications 
	fig, (ax1, ax2) = plt.subplots(nrows=2)
	ax1.plot(y,  (zuntilt-min(zuntilt))*1e-4, color='navy' , linestyle= 'solid')
	ax2.plot( y, zlevel*1e-4, color='tomato', label='rms:    ' +str(round(rms, 6))+'\nmean: '+  "{:.3e}".format(avg))
	
	ax2.legend()
	
	ax1.set_xticks(range(0,  max(y)+20, 500))
	ax2.set_xticks(range(0, max(y)+20, 500))
	ax1.set_yticks(range(0, 35, 10))
	ax2.set_yticks(range(-10, 2, 2))

	ax1.set_title(title + ': Leveled data')
	ax2.set_title(title + ': Major curvature removed')

	ax1.set_ylabel("height \u03BCm")
	ax2.set_ylabel("height \u03BCm")
	ax1.set_xlabel("scan length \u03BCm")
	ax2.set_xlabel("scan length \u03BCm")

	fig.tight_layout()
	plt.show()
	return

def main():
	name = "MoNbTaWV_2D_karkea.txt"
	title = "Rough sample (A)"
	y, z = data_read(name)
	
	for degree in range(1, 10):
		if degree ==1:													#correct tilt
			p =  np.polyfit([y[0], y[-1]],[z[0], z[-1]], 1)
			levelingarray = np.asarray(p[0]*y[0:]+p[1])
			zuntilt = z-levelingarray
			zlevel = copy.deepcopy(zuntilt)
		else:															#correct curvature
			pi = poly(zlevel, y, degree)
			zlevel = zlevel - pi(y)

	rms = np.sqrt(np.mean((zlevel*1e-4)**2))							#calculate roughness parameters
	avg = np.mean((zlevel*1e-4))
	print("rms: \t %1.16f \nmean: \t %s" % (rms, "{:.3e}".format(avg)))

	make_plots(y, zuntilt, zlevel, avg, rms, title)			#plots
	

	
if __name__ == '__main__':
	main()
