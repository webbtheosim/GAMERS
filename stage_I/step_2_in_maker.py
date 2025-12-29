import numpy as np

from scipy.optimize import fsolve
from functools import partial

from pathlib import Path

import sys


def lk_from_k(k,W) :
    
    l_b = 0.965
    l_k0 = l_b*(2*k+np.exp(-2*k)-1)/(1-np.exp(-2*k)*(2*k+1))
    dl_kW = 0.3638*(1-np.tanh(-5.481*W**3-3.870*W+2.318))*(1+np.tanh(0.002*k**3-0.469*k+0.214))
    
    return l_k0 + dl_kW

def k_solve(k_prime,k,W) :
    
    l_b = 0.965
    l_k0 = l_b*(2*k+np.exp(-2*k)-1)/(1-np.exp(-2*k)*(2*k+1))
    dl_k = 0.7276*(1+np.tanh(0.002*k**3-0.469*k+0.214))
    l_k0W = l_b*(2*k_prime+np.exp(-2*k_prime)-1)/(1-np.exp(-2*k_prime)*(2*k_prime+1))
    dl_kW = 0.3638*(1-np.tanh(-5.481*W**3-3.870*W+2.318))*(1+np.tanh(0.002*k_prime**3-0.469*k_prime+0.214))
    
    return l_k0 + dl_k - l_k0W - dl_kW
    
def WCA(x) :
    return 4*(1/x**12-1/x**6+1/4)

def dWCA(x) :
    return -4*(12/x**13-6/x**7)

def pair_table_writer(file,rc) :
    
    N = 10000
    
    xs = np.linspace(0.0001,1.1224620483,N)
    Us = WCA(xs)
    Fs = -dWCA(xs)
    
    for i,x in enumerate(xs) :
        if x < rc :
            Us[i] = WCA(rc) + (x-rc)*dWCA(rc)
            Fs[i] = -dWCA(rc)
    
    f = open(file,'w')
    
    f.write('Nonbonded\nN {}\n\n'.format(N))
    
    for i in range(N) :
        f.write('{} {} {} {}\n'.format(i+1,xs[i],Us[i],Fs[i]))
    
    f.close()

def pair_table_writer_zero_pressure(file) :
    
    rc = 2**(1/6)
    rca = 2**(1/2)
    N = 10000
    
    xs = np.linspace(0.0001,rca,N)
    Us = WCA(xs) - 0.5145
    Fs = -dWCA(xs)
    
    for i,x in enumerate(xs) :
        if rc < x <= rca:
            Us[i] = 0.5145*(np.cos(np.pi*(x/rc)**2)-1)
            Fs[i] = 0.5145*2*np.pi*x/rc**2*np.sin(np.pi*(x/rc)**2)
    
    f = open(file,'w')
    
    f.write('Nonbonded\nN {}\n\n'.format(N))
    
    for i in range(N) :
        f.write('{} {} {} {}\n'.format(i+1,xs[i],Us[i],Fs[i]))
    
    f.close()

with open(Path(__file__).resolve().parent.parent / 'GAMERS.input') as input:
    lines = input.readlines()
    inputs = {}
    for line in lines[:-1]:
        inputs[line.split(':')[0]] = line.strip().split(':')[1]

path = inputs['output_directory']

f_in = path + '/stage_I/step_2_inputs/step_2_base.in'
f_out = path + '/stage_I/step_2_inputs/step_2.in'
f_tab = path + '/stage_I/step_2_inputs/WCA_soft_U0_{}.table'

f = open(path + '/stage_I/step_1.mapping')
f.readline()
info = f.readline().split()
N_b = int(info[0])
k = float(info[1])
L = float(info[6])/float(info[2])
l_b = 0.965
f.close()

f = open(f_in)
lines = f.readlines()
f.close()

lines[0] = '# VARIABLES\n'
lines[1] = 'variable        data_name           index    ' + path + '/stage_I/step_2_inputs/step_2.data\n'
lines[2] = 'variable        final               index    ' + path + '/stage_I/step_2.end\n'
lines[3] = 'variable        thermo_freq         index    100000\n'
lines[4] = 'variable        time_step           index    0.01\n'
lines[5] = 'variable        soft_steps          index    1000000\n'
lines[6] = 'variable        full_steps          index    1000000\n'
lines[7] = 'variable        seed_val            index    {}\n'.format(np.random.randint(10**6))

for l, line in enumerate(lines) :

    if 'PYTHON pair' in line :
        
        try :
            U0 = int(line.split()[2])
        except :
            U0 = float(line.split()[2])
            
        rc = 26**(1/6)/(7+np.sqrt(36+13*U0))**(1/6)
        
        pair_table_writer(f_tab.format(U0),rc)
        
        lines[l] = 'pair_coeff 1 1 {} Nonbonded\nspecial_bonds fene\n'.format(f_tab.format(U0))

    if 'PYTHON angle' in line :
        
        try :
            U0 = int(line.split()[2])
        except :
            U0 = float(line.split()[2])
            
        W = U0**(1/6)/(1+U0**(1/6))
        
        k_prime_solve = partial(k_solve,k=k,W=W)
        k_prime = fsolve(k_prime_solve,k,maxfev=10**6)

        if N_b > 2:
            lines[l] = 'angle_coeff 1 {}\n'.format(k_prime[0])
        else:
            lines[l] = ''

    if 'bond_coeff' in line and N_b < 2:
        lines[l] = ''
    
f = open(f_out,'w')
for line in lines :
    f.write(line)
f.close()
