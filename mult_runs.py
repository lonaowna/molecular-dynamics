import argon_Ludwig_cut_off as argon_Ludwig
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

rhos = [0.88]

cvd = [] # The d`s mean these are arrays for the dictionary
Pd = []
difd = []
rhod = []
Td = []
U_potd = []

for i in range(0,1):

	T = 1.0
	rho = rhos[i]
	num_particles = 864
	n_iter = 1000
	n_iter_init = 200
	n_iter_cut = 15
	dt = 0.004
	n_t = 1
	bin_count = 300
	bin_size = 0.1
	N = num_particles
	a = 1 #How many times to iterate over one rho
	L = (num_particles/rho)**(1/3)
	rcut = 3.2
	rrms = 0

	rmax = rcut+rrms

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

	def storvar(vardict):
		f = open('var_comp_rho{0}_T{1}_rmax{2}_test2.txt'.format(rho, T, rmax), 'wb')
		pickle.dump(vardict,f,)
		f.close()
		return

	def Savefig(bin_size, bin_count, bins, rho, T, Vtot, Ktot, Etot, Tcurrent, v_sum, meandxlen):
		
		plt.figure(1)
		xvalues = numpy.array(range(0,bin_count))*bin_size
		plt.plot(xvalues,bins)
		plt.xlabel("r (sigma)")
		plt.ylabel("n")
		plt.title('rho={0} Temp={1}'.format(rho, T))
		plt.savefig("Correlation_rho_{0}_Temp_{1}_test2.png".format(rho, T))
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
		plt.savefig('All_rho_{0}_T_{1}_test2.png'.format(rho,T),bbox_inches='tight')
		plt.clf()
		
		plt.figure(3)
		plt.plot(range(0,n_iter),meandxlen, label='mean diffusion')
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.xlabel('Number of iterations')
		plt.title('rho={0} T={1}'.format(rho, T))
		plt.savefig('Diff_rho_{0}_T_{1}_test2.png'.format(rho,T),bbox_inches='tight')
		plt.clf()
		return

	### Start of program ###
	########################
	########################


	for j in range(0,a):

		CvN, Pbeta, bins, Vtot, Ktot, Etot, Tcurrent, meandxlen, v_sum = argon_Ludwig.mainf(T, rho, num_particles, n_iter, n_iter_init, dt, bin_count, bin_size, rcut, rrms, n_iter_cut)
		cvd.append(CvN)
		Pd.append(Pbeta)
		difd.append(meandxlen[n_iter-1])
		rhod.append(rho)
		Td.append(T)
		U_pot = numpy.mean(Vtot[numpy.int(n_iter*0.9):n_iter])/num_particles
		U_potd.append(U_pot)
		print('iteration a=',j)


	vardict = {'rho': rhod, 'T': Td, 'U_pot': U_potd,'cv': cvd, 'Diff_length': difd, 'Pressure':Pd}
	Savefig(bin_size, bin_count, bins, rho, T, Vtot, Ktot, Etot, Tcurrent, v_sum, meandxlen)
	storvar(vardict)



# meandx = meandxlen[n_iter-1]
# U_pot = numpy.mean(Vtot[numpy.int(n_iter*0.9):n_iter])/num_particles

# varnames = ('Cv', 'Upot', 'meandx', 'Tinit', 'Tend', 'Pbeta', 'num_particles', 'n_iter', 'n_iter_init', 'dt')





# print("Cv/N:", CvN)
# print('Diffusion length: ', meandx)
# print('num_part= ', num_particles)
# print('Temperature= ', T)
# print('rho= ', rho)
# print('P_beta= ',Pbeta)
# print('U_pot= ', U_pot)

# plt.figure(1)
# plt.plot(range(0,n_iter),Vtot/N)
# plt.plot(range(0,n_iter),Ktot/N)
# plt.plot(range(0,n_iter),Etot/N)
# plt.plot(range(0,n_iter),Tcurrent)
# plt.plot(range(0,n_iter),v_sum)
# plt.plot(range(0,n_iter),meandxlen)
# plt.legend(['V', 'K', 'E', 'T', 'v_sum', 'mean diffusion length'], loc='lower right')

# plt.figure(2)
# xvalues = numpy.array(range(0,bin_count))*bin_size
# plt.plot(xvalues,bins)
# plt.xlabel("r (sigma)")
# plt.ylabel("n")
# plt.show()