import argon
import math
import scipy.stats
from scipy.optimize import curve_fit
import random
import time
import numpy
import pickle
import matplotlib.pyplot as plt
from numba import jit

def storvar(vardict):
	f = open('output/var_comp_rho{0}_T{1}_rm{2}_{3}.txt'.format(rho, T, r_m, n_run), 'wb')
	pickle.dump(vardict,f,)
	f.close()
	return

def savefig(bin_size, bin_count, bins, rho, T, Vmean, Kmean, Emean, Tcurrent, v_sum, meandiff2, n_run):
	plt.figure(1)
	xvalues = numpy.array(range(0,bin_count))*bin_size
	plt.plot(xvalues,bins)
	plt.gca().set_xlim([0,L/2])
	plt.xlabel("r (sigma)")
	plt.ylabel("n")
	plt.title('rho={0} Temp={1}'.format(rho, T))
	plt.savefig("output/Correlation_rho{0}_T{1}_rm{2}_{3}.png".format(rho, T, r_m,n_run))
	plt.clf()

	plt.figure(2)
	plt.plot(range(0,n_iter),Vmean, label='V')
	plt.plot(range(0,n_iter),Kmean, label='K')
	plt.plot(range(0,n_iter),Emean, label='E_tot')
	plt.plot(range(0,n_iter),Tcurrent, label='T')
	plt.plot(range(0,n_iter),v_sum, label='v_sum')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlabel('Number of iterations')
	plt.title('rho={0} T={1}'.format(rho, T))
	plt.savefig('output/All_rho{0}_T{1}_rm{2}_{3}.png'.format(rho,T,r_m,n_run),bbox_inches='tight')
	plt.clf()
	
	plt.figure(3)
	plt.plot(range(0,n_iter),meandiff2, label='mean diffusion')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlabel('Number of iterations')
	plt.title('rho={0} T={1}'.format(rho, T))
	plt.savefig('output/Diff_rho{0}_T{1}_rm{2}_{3}.png'.format(rho,T,r_m, n_run),bbox_inches='tight')
	plt.clf()
	return

def make_table(vardict):
	a = vardict['rho']
	b = vardict['T']
	c = vardict['U_pot']
	d = vardict['cv']
	e = vardict['Diff_length']
	f = vardict['Pressure']
	names = ('Density', 'Temperature', 'Potential', 'Cv', 'Diffusion', 'Pressure')
	t = Table ([a, b, c, d, e, f], names=('Density', 'Temperature', 'Potential', 'Cv', 'Diffusion', 'Pressure'))
	return t

cvd = [] # The d`s mean these are arrays for the dictionary
Pd = []
difd = []
rhod = []
Td = []
U_potd = []
p_vard = []

diffusion_graphs = []

for rho,T in [[0.3,3], [0.88,1], [1.8,0.5]]:
	num_particles = 864
	n_iter = 1040
	n_iter_init = 40
	n_iter_cut = 15
	n_pres = 100
	n_run =2
	dt = 0.004
	n_t = 1
	bin_count = 300
	bin_size = 0.1
	L = (num_particles/rho)**(1/3)
	v_rms = math.sqrt(T)
	r_v = 2.5
	n_repeat = 1
	#r_m = r_v + v_rms*n_iter_cut*dt
	r_m = 3.5
	print("r_v=",r_v," r_m=",r_m)

	for i in range(0,n_repeat):
		x = argon.initial_positions(num_particles, L)
		v = argon.initial_velocities(num_particles, T)
		a = argon.initial_accelerations(num_particles)

		r_all = argon.interacting_particles(x, num_particles, L, r_m)
		CvN, meandiff2, Pbeta, bins, Vtot, Ktot, Etot, Tcurrent, v_sum, P_var = argon.run(num_particles, L, T, rho, x, v, a, dt, n_t, r_v, r_m, n_iter, n_iter_init, n_iter_cut, bin_count, bin_size, r_all, n_pres)

		print('CvN=', CvN)
		print('Pbeta= ',Pbeta)
		print('P_var=',P_var)
		p_vard.append(P_var)
		cvd.append(CvN)
		Pd.append(Pbeta)
		difd.append(meandiff2[n_iter-1])
		rhod.append(rho)
		Td.append(T)
		U_pot = numpy.mean(Vtot[numpy.int(n_iter*0.9):n_iter])/num_particles
		U_potd.append(U_pot)

	vardict = {'rho': rhod, 'T': Td, 'U_pot': U_potd,'cv': cvd, 'Diff_length': difd, 'Pressure':Pd, 'P_var': p_vard}
	savefig(bin_size, bin_count, bins, rho, T, Vtot/num_particles, Ktot/num_particles, Etot/num_particles, Tcurrent, v_sum, meandiff2, n_run)
	storvar(vardict)

	time = range(0,(n_iter-n_iter_init+1))
	time = [x * dt for x in time]

	diffusion_graphs.append({'T':T, 'rho':rho, 'x':time, 'y':meandiff2[n_iter_init-1:n_iter]})

plt.figure(4)
for diffusion_graph in diffusion_graphs:
	def diff_fitfunc(x, D):
		return D*x
	popt, pcov = curve_fit(diff_fitfunc, diffusion_graph['x'], diffusion_graph['y'])
	plt.plot(diffusion_graph['x'], diffusion_graph['y'], label='T={} rho={} D={}+-{}'.format(diffusion_graph['T'],diffusion_graph['rho'],popt[0],pcov[0]))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('time')
plt.ylabel('<x^2>')
plt.title('diffusion')
plt.savefig('diffusion.png',bbox_inches='tight')
plt.clf()
