import argon
import math
import scipy.stats
import random
import time
import numpy
import pickle
import matplotlib.pyplot as plt
from numba import jit
from scipy.optimize import curve_fit

def savefig(bin_size, bin_count, bins, rho, T, Vmean, Kmean, Emean, Tcurrent, v_sum, meandiff2):
	plt.figure(1)
	xvalues = numpy.array(range(0,bin_count))*bin_size
	plt.plot(xvalues,bins)
	plt.gca().set_xlim([0,L/2])
	plt.xlabel("r (sigma)")
	plt.ylabel("n")
	plt.title('rho={0} Temp={1}'.format(rho, T))
	plt.savefig("output/Correlation_rho{0}_T{1}_rm{2}.png".format(rho, T, r_m))
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
	plt.savefig('output/All_rho{0}_T{1}_rm{2}.png'.format(rho, T, r_m), bbox_inches='tight')
	plt.clf()
	
	plt.figure(3)
	plt.plot(range(0,n_iter),meandiff2, label='mean diffusion')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlabel('Number of iterations')
	plt.title('rho={0} T={1}'.format(rho, T))
	plt.savefig('output/Diff_rho{0}_T{1}_rm{2}.png'.format(rho, T, r_m), bbox_inches='tight')
	plt.clf()
	return

###

diffusion_graphs = []
for rho,T in [[0.3,3], [0.88,1], [1.8,0.5]]:
	n_repeat = 1
	num_particles = 864
	dt = 0.004
	n_t = 1
	n_iter_init = 20
	n_iter = n_iter_init + 25
	bin_count = 300
	bin_size = 0.1
	L = (num_particles/rho)**(1/3)
	v_rms = math.sqrt(T)
	r_v = 2.5
	#r_m = r_v + v_rms*interactcut_interval*dt
	r_m = 3.5
	print("r_v=",r_v," r_m=",r_m)

	x = argon.initial_positions(num_particles, L)
	v = argon.initial_velocities(num_particles, T)
	a = argon.initial_accelerations(num_particles)

	r_all = argon.interacting_particles(x, num_particles, L, r_m)
	CvN, meandiff2, P, bins, Vtot, Ktot, Etot, Tcurrent, v_sum = argon.run(num_particles, L, T, rho, x, v, a, dt, n_t, r_v, r_m, n_iter, n_iter_init, bin_count, bin_size, r_all)

	savefig(bin_size, bin_count, bins, rho, T, Vtot/num_particles, Ktot/num_particles, Etot/num_particles, Tcurrent, v_sum, meandiff2)

	time = range(0,(n_iter-n_iter_init+1))
	time = [x * dt for x in time]

	diffusion_graphs.append({'T':T, 'rho':rho, 'x':time, 'y':meandiff2[n_iter_init-1:n_iter]})

plt.figure(4)
for diffusion_graph in diffusion_graphs:
	def diff_fitfunc(x, D):
		return D*x
	popt, pcov = curve_fit(diff_fitfunc, diffusion_graph['x'], diffusion_graph['y'])
	plt.plot(diffusion_graph['x'], diffusion_graph['y'], label='T={} rho={} D={:.2f}±{:.1e}'.format(diffusion_graph['T'],diffusion_graph['rho'],float(popt[0]),float(pcov[0])))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('time')
plt.ylabel('$<x^2>$')
plt.title('diffusion')
plt.savefig('diffusion.png',bbox_inches='tight')
plt.clf()

###

pressure_graphs = []
for T in [0.5, 1, 1.5, 2.0]:
	xvalues = []
	yvalues = []
	for rho in [0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
		n_repeat = 1
		num_particles = 864
		dt = 0.004
		n_t = 1
		n_iter_init = 100
		n_iter = n_iter_init + 200
		bin_count = 300
		bin_size = 0.1
		L = (num_particles/rho)**(1/3)
		v_rms = math.sqrt(T)
		r_v = 2.5
		#r_m = r_v + v_rms*interactcut_interval*dt
		r_m = 3.5
		print("r_v=",r_v," r_m=",r_m)

		x = argon.initial_positions(num_particles, L)
		v = argon.initial_velocities(num_particles, T)
		a = argon.initial_accelerations(num_particles)

		r_all = argon.interacting_particles(x, num_particles, L, r_m)
		CvN, meandiff2, P, bins, Vtot, Ktot, Etot, Tcurrent, v_sum = argon.run(num_particles, L, T, rho, x, v, a, dt, n_t, r_v, r_m, n_iter, n_iter_init, bin_count, bin_size, r_all)
		xvalues.append(rho)
		yvalues.append(P)

	pressure_graphs.append({'T':T, 'x':xvalues, 'y':yvalues})

plt.figure(5)
for pressure_graph in pressure_graphs:
	specific_volume = [1/x for x in pressure_graph['x']]
	compressibility = [x/pressure_graph['T'] for x in pressure_graph['y']]
	plt.plot(specific_volume, compressibility, label='T={}'.format(pressure_graph['T']))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('specific volume')
plt.ylabel('P/kT')
plt.title('compressibility')
plt.savefig('compressibility.png',bbox_inches='tight')
plt.clf()