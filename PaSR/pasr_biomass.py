#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scikits.odes import ode
import matplotlib.pyplot as plt
import random
import cantera as ct
from numpy import random as np_random
import pandas as pd
mech = './mechanism_2008.cti'
# import the models for gas
gas1 = ct.Solution(mech)
T0 = 773  # Kelvin
p0 = ct.one_atm  # Pa
gas1.TPY = T0, p0, "CELL(s):0.45, HCE(s):0.33, LIG_C(s):0.06, LIG_H(s):0.05, LIG_O(s):0.11"
gas1()
a = 9
b = -6
num = (a-b)*10
print("limits:",a,b,num)
# calculate the site fractions of surface species at the entrance of the tube at steady state
N1 = gas1.n_species  # number of gas species
print("\nNumber of elements is: "+str(gas1.n_elements))
print(gas1.element_names)
print("\nNumber of species is: "+str(gas1.n_species))
print(gas1.species_names)
print("\nNumber of reactions is: "+str(gas1.n_reactions))

######################################## CVODE solver ###################################################################
def rhseqn(t, vec, vecp):
    """ we create the equations for the problem
        vec = [Yk]
        vecp = [dYkdt]
    """
    # temporary variables
    Y = vec[0:N1]  # vector of mass fractions of all species
    gas1.set_unnormalized_mass_fractions(Y)   
    rho = gas1.density
    wdot_g = gas1.net_production_rates  # homogeneous production rate of species
    W_g = gas1.molecular_weights  # vector of molecular weight of species
    # conservation of species
    for k in range(N1):
        vecp[k] = wdot_g[k]*W_g[k]/rho
################### ODE Solver ###########################
vec0 = np.hstack((gas1.Y)) # initial mass fraction
print(vec0)
solver = ode(
    'cvode',
    rhseqn, 
    atol=1e-08,  # absolute tolerance for solution
    rtol=1e-08,  # relative tolerance for solution
    max_steps=5000,
    old_api=False
)
#
time = np.logspace(b, a, num)
solution = solver.solve(time, vec0)
if solution.errors.t:
    print ('Error: ', solution.message, 'Error at time', solution.errors.t)

mass_fraction = np.zeros((num,N1))
for i in range(0,num):
    for j in range(0,N1):
        mass_fraction[i,j] = solution.values.y[i,j]
print("#################### Solving devolatilization is finished ####################")
print("#################### Preparing Input data for PFR ####################")
print("#################### Calculating mass flow rate ####################")
m_bio = 0.2778 # g/s
m_air = 0.1662 # g/s
m_dot = m_bio * mass_fraction[num-1,0:N1] # g/s
total_m_dot = np.zeros((num))
total_m_dot = np.sum(m_dot) # it should be zero. mass conservation
print(total_m_dot) # real total m dot
for j, name in enumerate(gas1.species_names):
     if (name.find('(s)') != -1):
          total_m_dot = total_m_dot - m_dot[j] # exclude what ever is not gas virgin and active (C, H, L) and G_*
print(total_m_dot) # total m dot of gases
total_m_dot_PFR = total_m_dot + m_air
print(total_m_dot_PFR)
print("#################### Calculating inlet mass fraction of PFR ####################")
#
m_O2 = 0.2*m_air
m_N2 = m_air - m_O2
#
inlet_PFR_Y = np.zeros((N1))
inlet_PFR_Y[0:N1] = m_dot[0:N1]/total_m_dot_PFR
inlet_PFR_Y_N2 = m_N2 / total_m_dot_PFR
inlet_PFR_Y_O2 = m_O2 / total_m_dot_PFR
gas_devol_names = gas1.species_names
print(gas_devol_names)
for i, name in enumerate(gas_devol_names):
	if (name == "Glyoxal"):
		gas_devol_names[i] = "C2H2O2"
	if (name == "HAA"):
		gas_devol_names[i] = "C2H4O2"
	if (name == "CH3CHO"):
		gas_devol_names[i] = "C2H4O"
	if (name == "HMFU"):
		gas_devol_names[i] = "C6H6O3"
	if (name == "Char(s)"):
		gas_devol_names[i] = "CSOLID(s)"
	if (name == "LVG"):
		gas_devol_names[i] = "C6H10O5"
	if (name == "XYLOSE"):
		gas_devol_names[i] = "C5H8O4"
	if (name == "pCoumaryl"):
		gas_devol_names[i] = "C9H10O2"
	if (name == "Phenol"):
		gas_devol_names[i] = "C6H5OH"
print(gas_devol_names)
Y_PFR = ""
for j, name in enumerate(gas_devol_names):
	if (name.find('(s)') == -1):
		Y_PFR = Y_PFR + gas_devol_names[j]+":"+str(inlet_PFR_Y[j])+", "
Y_PFR = Y_PFR + "N2:"+str(inlet_PFR_Y_N2)+", "
Y_PFR = Y_PFR + "O2:"+str(inlet_PFR_Y_O2)
mech2 = './biomass.cti'
print("#################### Reading CRECK Data ####################")
gas2 = ct.Solution(mech2)
gas2()
print("\nNumber of species is: "+str(gas2.n_species))
print(gas2.species_names)
print("\nNumber of reactions is: "+str(gas2.n_reactions))
T0 = 773  # Kelvin # should be between 425-500 C or 698-773 K
p0 = ct.one_atm  # atm
gas2.TPY = T0, p0, Y_PFR
mu = np.zeros((gas2.n_species+1))
mu[0:gas2.n_species]=gas2.Y
mu[gas2.n_species]=T0
print("#################### Input data are ready PFR ####################")
#
addRxn = 1
addInflow = 0
addMixing = 1
Ns = len(mu)
num_samples = 100
tau_m = 0.1
Q = 1.6667e-06
D = 3.81e-02  # diameter of the tube [m]
Ac = np.pi * D**2/4
length = 32.2e-01
numberOfPaSR = 200
u0 =  (total_m_dot_PFR * 1e-3 )/ (gas2.density * Ac)
C_phi = 2
#
# covariance matrix.
r = np.identity(Ns)
lower = 0
upper = 1
def beta_mean(lwr, high, mean, kappa):
    offset = lwr
    scale = high - lwr
    meanAct = (mean - offset)/scale
    beta = (kappa/meanAct) - kappa
    if beta <= 0:
        scale = 2000 - lwr
        meanAct = (mean - offset)/scale
        beta = (kappa/meanAct) - kappa
    return offset + scale * np_random.beta(kappa, beta, num_samples)
#
a = 0
b = 1
xx = np.zeros((Ns,num_samples))
for bb  in range(Ns):
    xx[bb,:] = beta_mean(a,b,mu[bb],10000)

y = xx 
y1 = y
t0 = 0
t_total = length / u0
dt = t_total / numberOfPaSR
tau = dt
t_iteratre = (np.arange(numberOfPaSR) + 1) * dt
delta_t = dt
z1 = np.zeros_like(t_iteratre)
u1 = np.zeros_like(t_iteratre)
mean_s = 0
y_mixing = np.zeros(num_samples) # allocate psi(t) for mixing
#
# operator splitting 
nk = 0
mass_frac = np.zeros((Ns,len(t_iteratre)+1))
moments0 = np.zeros((Ns,len(t_iteratre)+1))
moments2 = np.zeros((Ns,len(t_iteratre)+1))
moments3 = np.zeros((Ns,len(t_iteratre)+1))
my_time = np.zeros((len(t_iteratre)+1))

for az in range(Ns):
    mass_frac[az,0] = np.sum(xx[az,:])/num_samples
    moments0[az,0] =  np.sum(np.power(xx[az,:],0))/num_samples
    moments2[az,0] =  np.sum(np.power(xx[az,:],2))/num_samples
    moments3[az,0] =  np.sum(np.power(xx[az,:],3))/num_samples   
def updateMixing(species,new_y):
            mean_s = sum(new_y[species,:]) / num_samples
            #print(mean_s)
            B = C_phi / (2*tau_m)
            for j in range (num_samples):
                y_mixing[j] = new_y[species,j]*np.exp(-B*delta_t) + (1-np.exp(-B*delta_t))*mean_s
            return y_mixing
def updateInflow(y2):       
        Nreplaced1 = np.ceil(num_samples*(delta_t/tau))
        Nreplaced = int(Nreplaced1)
        ind = np.zeros(Nreplaced)
        yreplaced = random.sample(list(enumerate(y2)),Nreplaced)
        for k in range(Nreplaced):
            ind[k] = yreplaced[k][0]
        return ind
def UpdateRxn(species_rxn,to):
        inlet_PFR_Y = species_rxn
        Y_PFR = ""
        nScalars = np.size(gas2.Y)
        Y_PFR = inlet_PFR_Y[0:nScalars]
        # initial conditions 
        T0 = species_rxn[-1]  
        p0 = ct.one_atm  # atm
        gas2.TPY = T0, p0, Y_PFR
        r = ct.IdealGasConstPressureReactor(gas2)
        sim = ct.ReactorNet([r])
        sim.verbose = False
        t_end = to
        states = ct.SolutionArray(gas2, extra=['t'])
        sim.set_initial_time(t0-delta_t)
        while sim.time < t_end:
            sim.advance(t_end)
            states.append(r.thermo.state, t=sim.time)  
        return states.Y, states.T
for n1, p in enumerate(t_iteratre):
    print(p)
    t0 = t0 + dt
    if (addMixing == 1) :
        for q in range(Ns):
             y1[q,:] = updateMixing(q,y1)
    if ( addInflow  == 1 ) :       
        for qf in range(Ns):
            myind = updateInflow(y1[qf,:])   
            for c in range(len(myind)):              
                cz = int(myind[c])   
                y1[qf,cz] = mu[qf]
    y2rxn = y1                 
    if ( addRxn  == 1 ) :   
        for qr in range(num_samples):
            aa, bb = UpdateRxn(y2rxn[:,qr],t0)
            y1[0:-1,qr]= aa
            y1[-1,qr] = bb    
    for az in range(Ns):
        mass_frac[az,nk+1] = np.sum(y1[az,:])/num_samples
    gas2.TPY = mass_frac[-1,n1+1], p0, mass_frac[0:-1,n1+1]
    u1[n1] = (total_m_dot_PFR *1e-3) / Ac / gas2.density
    z1[n1] = z1[n1 - 1] + u1[n1] * dt
    nk = nk+1  
    mu = mass_frac[:,nk]
    my_time[nk] = t0

y=y1
z1_new = np.zeros(len(z1)+1)
z1_new[1:] = z1

pd.DataFrame(my_time).to_csv("full_PaSR_time_ode15s_cantera_tau_m_01.csv", header=None, index=None)
pd.DataFrame(mass_frac).to_csv("full_massFractions_ode15s_cantera_tau_m_01.csv", header=None, index=None)
pd.DataFrame(z1_new).to_csv("full_distance_cantera_tau_m_01.csv", header=None, index=None)
