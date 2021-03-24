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

from scipy.special import wofz

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

	def pretty_data(self, data, min_wav=0, max_wav=4000, plot=False):
		"""
		Creates two sub 1D arrays which cover only a domain and range of interest. Can plot this desired region also (plot = True).

		Parameters
		-----------
			data : pd.DataFrame
			min_wav : int or float, The minimum wavenumber of interest. Default is 0.
			max_wav : int or float, The maximum wavenumber of interest. Default is 4000.


		Returns
		-----------
			x : 1D array, domain of interest
			y : 1D array, range of interest


		"""
		
		x = data[(data.Wavenumber >= min_wav) & (data.Wavenumber <= max_wav)]['Wavenumber']
		y = data[(data.Wavenumber >= min_wav) & (data.Wavenumber <= max_wav)]['Absorbance']

		self.x = x
		self.y = y

		if plot == True:
			plt.plot(x, y)

		return x, y

	@staticmethod
	def fun_gauss(x, amp,cen,sigma):
		"""
		Gaussian function evaluation.

		Parameters
		-----------
			x : 1D array, Wavenumbers
			amp : float, Amplitude of the function
			cen : float, Centroid of the function
			sigma : float, Standard deviation of the function


		Returns
		-----------
			y : 1D array, Gaussian function values


		"""
		
		return amp*(1/(sigma*(np.sqrt(2*np.pi))))*(np.exp((-1/2)*((x-cen)**2/(sigma**2))))

	@staticmethod
	def fun_lorentzian(x, amp, cen, gamma):
		"""
		Lorentzian function evaluation.

		Parameters
		-----------
			x : 1D array, Wavenumbers
			amp : float, Amplitude of the function
			cen : float, Centroid of the function
			gamma : float, specifies the half-width at half-maximum (HWHM), 2*gamma is full width at half maximum (FWHM). Gamma is also twice the IQR.


		Returns
		-----------
			y : 1D array, Lorentzian function values


		"""
		
		return amp*(1/(np.pi*gamma*(1+((x - cen)/gamma)^2)))

	@staticmethod
	def fun_voigt(x, amp, cen, sigma, gamma):
		"""
		Voigt function evaluation.

		Parameters
		-----------
			x : 1D array, Wavenumbers
			amp : float, Amplitude of the function
			cen : float, Centroid of the function
			sigma : float, specifies the standard deviation of the Gaussian function.
			gamma : float, specifies the half-width at half-maximum (HWHM) of the Lorentzian function, 2*gamma is full width at half maximum (FWHM). Gamma is also twice the IQR.


		Returns
		-----------
			y : 1D array, Voigt function values


		"""

		return amp*np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma/np.sqrt(2*np.pi)


	def fun(self, fun_name):
		"""
		Choose the type of function you want to fit the data.

		Parameters
		-----------
			fun_name : str, 'gaussian', 'lorentzian' or 'voigt'.


		"""

		if fun_name == 'gaussian':
			self.fun_name = spec_deconv.fun_gauss

		elif fun_name == 'lorentzian':
			self.fun_name = spec_deconv.fun_lorentzian

		elif fun_name == 'voigt':
			self.fun_name = spec_deconv.fun_voigt
		else:
			print("Check spelling.")

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

		if self.fun_name == spec_deconv.fun_voigt:
			for i in range(self.num_peaks):
				params.add_many((f'amp_{i+1}',   0.1,    True,  0,    200,   None),
				(f'cen_{i+1}',   centroids[i],   True,  centroids[i]*lower, centroids[i]*upper, None),
				(f'sigma_{i+1}',   25,     True,  0,    1000,  None),
				(f'gamma_{i+1}',   25,     True,  0,    1000,  None))
			
		else:
			for i in range(self.num_peaks):
					params.add_many((f'amp_{i+1}',   0.1,    True,  0, None,  None), #was 0, 75
					(f'cen_{i+1}',   centroids[i],   True,  centroids[i]*lower, centroids[i]*upper, None), #was centroids[i]*lower, centroids[i]*upper
					(f'sigma_{i+1}',   25,     True,  0, None,  None)) #was 0, 310

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
	def _res(params, x, y, fun):
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

		if fun == spec_deconv.fun_voigt:
			for i in range(num_peaks):
				globals()[f"peak{i+1}"] = fun(x,params[f'amp_{i+1}'].value,params[f'cen_{i+1}'].value,params[f'sigma_{i+1}'].value, params[f'gamma_{i+1}'].value)
				model += globals()[f"peak{i+1}"]

		else:
			for i in range(num_peaks):
				globals()[f"peak{i+1}"] = fun(x,params[f'amp_{i+1}'].value,params[f'cen_{i+1}'].value,params[f'sigma_{i+1}'].value)
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
		all_peaks = []
		num_peaks = params['num_peaks'].value

		for i in range(num_peaks):
			globals()[f"peak{i+1}"] = spec_deconv.fun_gauss(self.x,params[f'amp_{i+1}'].value,params[f'cen_{i+1}'].value,params[f'sigma_{i+1}'].value)
			model += globals()[f"peak{i+1}"]
			all_peaks.append(globals()[f"peak{i+1}"])

		all_peaks.insert(0, model)
		self.model = model
	    
		for i in all_peaks:
			yield i


	def update_model(self, params):
		"""
		Updates the model based on parameters. Useful for post fitting.

		Parameters
		-----------
			params : parameters from lmfit module


		Returns
		-----------
			model : 2D array, model overarching peak based on convolution of sub-peaks.

		"""

		model = 0
		all_peaks = []
		num_peaks = params['num_peaks'].value

		for i in range(num_peaks):
			globals()[f"peak{i+1}"] = spec_deconv.fun_gauss(self.x,params[f'amp_{i+1}'].value,params[f'cen_{i+1}'].value,params[f'sigma_{i+1}'].value)
			model += globals()[f"peak{i+1}"]
	
		return model


	def fit(self, params, x, y, method = 'least_squares', results = False):
		"""
		Fits the raw data to the desired function - i.e. by minimising the residuals.

		Parameters
		-----------
			params : parameters from lmfit module
			x : 1D array, domain of interest
			y : 1D array, range of interest
			method : str, method for fitting. Default is 'least_squares'. Full list found in lmfit documentation.
			results : bool, Print results.


		Returns
		-----------
			fit : lmfit.parameter.Parameters, optimised parameters from fit.


		"""
		fit = lmfit.minimize(spec_deconv._res, params, method = method, args=(x,y, self.fun_name))

		self.model = spec_deconv.update_model(self, fit.params)

		if results == True:
			print(lmfit.report_fit(fit))
		
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
	
	def plot_all(self, params, save=False, name=None, opac = 0.5):
		"""
		Plots the raw data and model, with deconvoluted peaks also.

		Parameters
		-----------
			params : parameters from lmfit module. Ideally the optimised/fitted parameters.
			save : bool, Variable to determine if the figure will be saved. Default is false.
			name : str, Name if the figure is to be saved.
			opac : flat, Opacity in [0,1]. Default is 0.5.

		Returns
		-----------


		"""
		fig = plt.figure(figsize=(10,6))
		gs = gridspec.GridSpec(1,1)
		ax1 = fig.add_subplot(gs[0])
		col = ['green', 'yellow', 'blue', 'red', 'purple', 'orange', 'pink']*3

		ax1.plot(self.x, self.y, "ro")
		ax1.plot(self.x, self.model, 'k--')

		for i in range(params['num_peaks'].value):
			p = spec_deconv.fun_gauss(self.x,params[f'amp_{i+1}'].value,params[f'cen_{i+1}'].value,params[f'sigma_{i+1}'].value)
			ax1.plot(self.x, p, "g")
			ax1.fill_between(self.x, p.min(), p, facecolor=col[i], alpha=opac)

		fig.tight_layout()
		fig.suptitle(name, fontsize=16, y=1.05)
		ax1.set_ylabel("Absorbance",family="serif",  fontsize=12)
		ax1.set_xlabel("Wavenumber (cm$^{-1}$)",family="serif",  fontsize=12)

		if save == True:
			fig.savefig(f"{name}.png", format="png",dpi=1000)