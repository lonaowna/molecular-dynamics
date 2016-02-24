import argon
import math
import numpy
import scipy.stats
import random
import time
#import anim_md
import matplotlib.pyplot as plt
from numba import jit
import pickle
#from astropy.table import Table, Column

rhos = [0.88,0.8,0.7]

cvd = [] # The d`s mean these are arrays for the dictionary
Pd = []
difd = []
rhod = []
Td = []
U_potd = []
p_vard = []

for i in range(0,len(rhos)):

	T = 1
	rho = rhos[i]
	num_particles = 864
	n_iter = 1400
	n_iter_init = 400
	n_iter_cut = 15
	n_pres = 100
	n_run =2
	dt = 0.004
	n_t = 1
	bin_count = 300
	bin_size = 0.1
	N = num_particles
	L = (num_particles/rho)**(1/3)
	v_rms = math.sqrt(T)
	r_v = 2.5
	a = 1
	#r_m = r_v + v_rms*n_iter_cut*dt
	r_m = 3.5
	print("r_v=",r_v," r_m=",r_m)

	def storvar(vardict):
		f = open('var_comp_rho{0}_T{1}_rm{2}_{3}.txt'.format(rho, T, r_m, n_run), 'wb')
		pickle.dump(vardict,f,)
		f.close()
		return

	def Savefig(bin_size, bin_count, bins, rho, T, Vtot, Ktot, Etot, Tcurrent, v_sum, meandxlen, n_run):
		plt.figure(1)
		xvalues = numpy.array(range(0,bin_count))*bin_size
		plt.plot(xvalues,bins)
		plt.gca().set_xlim([0,L/2])
		plt.xlabel("r (sigma)")
		plt.ylabel("n")
		plt.title('rho={0} Temp={1}'.format(rho, T))
		plt.savefig("Correlation_rho{0}_T{1}_rm{2}_{3}.png".format(rho, T, r_m,n_run))
		plt.clf()

		plt.figure(2)
		plt.plot(range(0,n_iter),Vtot/N, label='V')
		plt.plot(range(0,n_iter),Ktot/N, label='K')
		plt.plot(range(0,n_iter),Etot/N, label='E_tot')
		plt.plot(range(0,n_iter),Tcurrent, label='T')
		plt.plot(range(0,n_iter),v_sum, label='v_sum')
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.xlabel('Number of iterations')
		plt.title('rho={0} T={1}'.format(rho, T))
		plt.savefig('All_rho{0}_T{1}_rm{2}_{3}.png'.format(rho,T,r_m,n_run),bbox_inches='tight')
		plt.clf()
		
		plt.figure(3)
		plt.plot(range(0,n_iter),meandxlen, label='mean diffusion')
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.xlabel('Number of iterations')
		plt.title('rho={0} T={1}'.format(rho, T))
		plt.savefig('Diff_rho{0}_T{1}_rm{2}_{3}.png'.format(rho,T,r_m, n_run),bbox_inches='tight')
		plt.clf()
		return

	### Start of program ###
	########################
	########################


	for i in range(0,a):
		CvN, Pbeta, bins, Vtot, Ktot, Etot, Tcurrent, meandxlen, v_sum, P_var = argon.mainf(T, rho, num_particles, n_iter, n_iter_init, dt, bin_count, bin_size, r_v, r_m, n_iter_cut, n_pres)
		print('CvN=', CvN)
		print('Pbeta= ',Pbeta)
		print('P_var=',P_var)
		p_vard.append(P_var)
		cvd.append(CvN)
		Pd.append(Pbeta)
		difd.append(meandxlen[n_iter-1])
		rhod.append(rho)
		Td.append(T)
		U_pot = numpy.mean(Vtot[numpy.int(n_iter*0.9):n_iter])/num_particles
		U_potd.append(U_pot)

	vardict = {'rho': rhod, 'T': Td, 'U_pot': U_potd,'cv': cvd, 'Diff_length': difd, 'Pressure':Pd, 'P_var': p_vard}
	Savefig(bin_size, bin_count, bins, rho, T, Vtot, Ktot, Etot, Tcurrent, v_sum, meandxlen, n_run)
	storvar(vardict)

	# def make_table(vardict):
	#     a = vardict['rho']
	#     b = vardict['T']
	#     c = vardict['U_pot']
	#     d = vardict['cv']
	#     e = vardict['Diff_length']
	#     f = vardict['Pressure']
	#     names = ('Density', 'Temperature', 'Potential', 'Cv', 'Diffusion', 'Pressure')
	#     t = Table ([a, b, c, d, e, f], names=('Density', 'Temperature', 'Potential', 'Cv', 'Diffusion', 'Pressure'))
	#     t
	#     return
