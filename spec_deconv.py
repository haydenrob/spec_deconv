# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:05:38 2021

@author: Hayden
"""

import lmfit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker

class spec_deconv:
	"""
	A simple Python script for the deconvolution of spectral peaks from FTIR or Raman scattering.

	--------------
	Parameters
		num_peaks : int, The number of peaks composing the spectral peak of interest
	"""
	
	def __init__(self, num_peaks):
		self.num_peaks = num_peaks

		print(f"Spectral deconvolution module loaded: {num_peaks} peaks.")
	
	def load(self, path):
		"""
		Loads in the data. Currently must be a 2D array in txt format, with the headers 'Wavenumber' and 'Absorbance'.

		Parameters
		-----------
			path : str, The file path where the data is stored.


		Returns
		-----------
			data : pd.DataFrame


		"""

		data = pd.DataFrame(np.genfromtxt(path, skip_header=1, names=['Wavenumber', 'Absorbance']))
		return data

	def pretty_data(self, data, wav_num, plot=False):
		"""
		Creates two sub 1D arrays which cover only a domain and range of interest. Can plot this desired region also (plot = True).

		Parameters
		-----------
			data : pd.DataFrame
			wav_num : int or float, The minimum wavenumber of interest.


		Returns
		-----------
			x : 1D array, domain of interest
			y : 1D array, range of interest


		"""
		
		x = data[data.Wavenumber > wav_num]['Wavenumber']
		y = data[data.Wavenumber > wav_num]['Absorbance']

		self.x = x
		self.y = y

		if plot == True:
			plt.plot(x, y)

		return x, y

	def fun_gauss(x, amp,cen,sig):
		"""
		Gaussian function evaluation

		Parameters
		-----------
			x : 1D array, Wavenumbers
			amp : float, Amplitude of the function
			cen : float, Centroid of the function
			sig : float, Standard deviation (sigma) of the function


		Returns
		-----------
			y : 1D array, Gaussian function values


		"""
		
		return amp*(1/(sig*(np.sqrt(2*np.pi))))*(np.exp((-1/2)*((x-cen)**2/(sig**2))))

	def auto_params(self, centroids, lower=0.99, upper=1.01):
		"""
		Automatically creates parameters for fitting based on the number of peaks requested and the centroids. 
		Can also change the lower and upper limits on the centroid parameters. - default is 0.99 and 1.01.

		Parameters
		-----------
			centroids : list, centroid locations for each sub-peak.
			lower : float, The lower bound on the centroid parameters. Default is 0.99.
			upper : float, The upper bound on the centroid parameters. Default is 1.01.


		Returns
		-----------
			params : parameters for lmfit module.


		"""
		
		params = lmfit.Parameters()
		params.add('num_peaks', value=self.num_peaks, vary=False)
		for i in range(self.num_peaks):
				params.add_many((f'amp_{i+1}',   0.1,    True,  0,    50,   None),
				(f'cen_{i+1}',   centroids[i],   True,  centroids[i]*lower, centroids[i]*upper, None),
				(f'sig_{i+1}',   25,     True,  0,    310,  None))

		return params

	def fix_centroid(self, params, will_vary = False):
		"""
		Automatically fixes or lets vary the centroids parameters. 

		Parameters
		-----------
			params : parameters for lmfit module.
			will_vary : bool, Will the centroid parameters be allowed to vary? Default is False.

		Returns
		-----------

		"""

		for i in range(self.num_peaks):
			params[f'cen_{i+1}'].vary = will_vary


	@staticmethod
	def res(params, x, y):
		"""
		Calculates the residuals between the model and the raw data.

		Parameters
		-----------
			params : parameters from lmfit module
			x : 1D array, domain of interest
			y : 1D array, range of interest


		Returns
		-----------
			res : 1D array, residuals from the model - y.


		"""

		model = 0
		all_peaks = []

		num_peaks = params['num_peaks'].value
		for i in range(num_peaks):
			globals()[f"peak{i+1}"] = spec_deconv.fun_gauss(x,params[f'amp_{i+1}'].value,params[f'cen_{i+1}'].value,params[f'sig_{i+1}'].value)
			model += globals()[f"peak{i+1}"]
    
		return abs(model - y)


	def update_peaks(self, params):
		"""
		Updates the model and sub-peaks based on parameters. Useful for post fitting.

		Parameters
		-----------
			params : parameters from lmfit module


		Returns
		-----------
			model : 2D array, model overarching peak based on convolution of sub-peaks.
			peak{n} : 2D array, deconvoluted nth peak from overarching peak.


		"""
		model = 0
		num_peaks = params['num_peaks'].value

		for i in range(num_peaks):
			globals()[f"peak{i+1}"] = spec_deconv.fun_gauss(self.x,params[f'amp_{i+1}'].value,params[f'cen_{i+1}'].value,params[f'sig_{i+1}'].value)
			model += globals()[f"peak{i+1}"]

		self.model = model
	    
		return model, peak1, peak2, peak3, peak4, peak5, peak6

	def fit(self, params, x, y, method = 'least_squares'):
		"""
		Fits the raw data to the desired function - i.e. by minimising the residuals.

		Parameters
		-----------
			params : parameters from lmfit module
			x : 1D array, domain of interest
			y : 1D array, range of interest
			method : str, method for fitting. Default is 'least_squares'. Full list found in lmfit documentation.


		Returns
		-----------
			fit : lmfit.parameter.Parameters, optimised parameters from fit.


		"""
		fit = lmfit.minimize(spec_deconv.res, params, method = method, args=(x,y))

		spec_deconv.update_peaks(self, fit.params)

		return fit

	def plot_model(self, model, save=False, name=None):
		"""
		Plots only the raw data and model.

		Parameters
		-----------
			model : 2D array, Array containing x and y coordinates of the model to plot.
			save : bool, Variable to determine if the figure will be saved. Default is false.
			name : str, Name if the figure is to be saved.

		Returns
		-----------


		"""
		fig = plt.figure(figsize=(10,6))
		gs = gridspec.GridSpec(1,1)
		ax1 = fig.add_subplot(gs[0])

		ax1.plot(self.x, self.y, "ro")
		ax1.plot(self.x, model, 'k--')

		fig.tight_layout()
		ax1.set_ylabel("Absorbance",family="serif",  fontsize=12)
		ax1.set_xlabel("Wavenumber (cm$^{-1}$)",family="serif",  fontsize=12)

		if save == True:
			fig.savefig(f"{name}.png", format="png",dpi=1000)	
	
	def plot_all(self, params, save=False, name=None):
		"""
		Plots the raw data and model, with deconvoluted peaks also.

		Parameters
		-----------
			params : parameters from lmfit module. Ideally the optimised/fitted parameters.
			save : bool, Variable to determine if the figure will be saved. Default is false.
			name : str, Name if the figure is to be saved.

		Returns
		-----------


		"""
		fig = plt.figure(figsize=(10,6))
		gs = gridspec.GridSpec(1,1)
		ax1 = fig.add_subplot(gs[0])
		col = ['green', 'yellow', 'blue', 'red', 'purple', 'orange', 'pink']

		ax1.plot(self.x, self.y, "ro")
		ax1.plot(self.x, self.model, 'k--')

		for i in range(self.num_peaks):
			p = spec_deconv.fun_gauss(self.x,params[f'amp_{i+1}'].value,params[f'cen_{i+1}'].value,params[f'sig_{i+1}'].value)
			ax1.plot(self.x, p, "g")
			ax1.fill_between(self.x, p.min(), p, facecolor=col[i], alpha=0.5)

		fig.tight_layout()
		ax1.set_ylabel("Absorbance",family="serif",  fontsize=12)
		ax1.set_xlabel("Wavenumber (cm$^{-1}$)",family="serif",  fontsize=12)

		if save == True:
			fig.savefig(f"{name}.png", format="png",dpi=1000)