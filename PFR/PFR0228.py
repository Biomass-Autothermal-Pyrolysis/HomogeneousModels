# PFR model implementation
# Copyright (C) 2022 Aziz D. Ilgun
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from scikits.odes import ode
import cantera as ct
############################### DEVOLATILIZATION   ##################################################################
############################### initial conditions ##################################################################
# input file containing the reaction mechanism
# ck2cti --input=mechanism_2008.cki --thermo=therm.dat --transport=tran.dat
# Reference : 
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
# calculate the site fractions of surface species at the entrance of the tube at steady state
N1 = gas1.n_species  # number of gas species
######################################## CVODE solver ###################################################################
def rhseqn(t, vec, vecp):
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
solver = ode(
    'cvode',
    rhseqn, 
    atol=1e-8,  # absolute tolerance for solution
    rtol=1e-8,  # relative tolerance for solution
    max_steps=5000,
    old_api=False
)
time = np.logspace(b, a, num)
solution = solver.solve(time, vec0)
mass_fraction = np.zeros((num,N1))
for i in range(0,num):
    for j in range(0,N1):
        mass_fraction[i,j] = solution.values.y[i,j]
m_bio = 0.2778 # g/s
m_air = 0.1662 # g/s
m_dot = m_bio * mass_fraction[num-1,0:N1] # g/s
total_m_dot = np.zeros((num))
total_m_dot = np.sum(m_dot) # it should be zero. mass conservation
for j, name in enumerate(gas1.species_names):
     if (name.find('(s)') != -1):
          total_m_dot = total_m_dot - m_dot[j] # exclude what ever is not gas virgin and active (C, H, L) and G_*
total_m_dot_PFR = total_m_dot + m_air
m_O2 = 0.2*m_air
m_N2 = m_air - m_O2
inlet_PFR_Y = np.zeros((N1))
inlet_PFR_Y[0:N1] = m_dot[0:N1]/total_m_dot_PFR
inlet_PFR_Y_N2 = m_N2 / total_m_dot_PFR
inlet_PFR_Y_O2 = m_O2 / total_m_dot_PFR
gas_devol_names = gas1.species_names
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
# import the gas phase reactions mechanism
mech2 = './biomass.cti'
gas2 = ct.Solution(mech2)
gas2()
Y_PFR = ""
for j, name in enumerate(gas_devol_names):
	if (name.find('(s)') == -1):
		Y_PFR = Y_PFR + gas_devol_names[j]+":"+str(inlet_PFR_Y[j])+", "
Y_PFR = Y_PFR + "N2:"+str(inlet_PFR_Y_N2)+", "
Y_PFR = Y_PFR + "O2:"+str(inlet_PFR_Y_O2)

length = 32.2e-01  # *approximate* PFR length [m]
D = 3.81e-02  # diameter of the tube [m]

area = np.pi * D**2/4

T0 = 773  # Kelvin # should be between 425-500 C or 698-773 K
p0 = ct.one_atm  # atm
gas2.TPY = T0, p0, Y_PFR
y0 = gas2.Y
u01 =  (total_m_dot_PFR * 1e-3 )/ (gas2.density * area)
mass_flow_rate1 = u01 * gas2.density * area

# create a new reactor
reactor = ct.FlowReactor(gas2)
reactor.mass_flow_rate = mass_flow_rate1/area
network = ct.ReactorNet([reactor])
network.set_max_time_step(0.0001)
network.rtol = 1e-8
network.atol = 1e-6
z2 = []
states1 = ct.SolutionArray(reactor.thermo)
while reactor.distance < length:
    z2.append(reactor.distance)
    network.step()
    states1.append(reactor.thermo.state)
z2 = np.array(z2)


import pandas as pd 
pd.DataFrame(z2).to_csv("PFR_distance.csv", header=None, index=None)
pd.DataFrame(states1.T).to_csv("PFR_temperature.csv", header=None, index=None)
pd.DataFrame(states1.Y).to_csv("PFR_massFractions.csv", header=None, index=None)
