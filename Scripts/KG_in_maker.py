import numpy as np

from scipy.optimize import fsolve
from functools import partial

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

mol = 'PEO'
path = '/scratch/gpfs/jm7732/Equil/PEO_Paper'
sys_typ = ''

if 'Film' in sys_typ:
    f_in = 'KG_film.in'
else:
    f_in = 'KG.in'
f_out = path + '/KG/Inputs/{}_KG_base.in'
f_tab = path + '/KG/Inputs/KG_U0_{}.table'

f = open(path + '/KG/{}_KG.mapping'.format(mol))
f.readline()
info = f.readline().split()
k = float(info[1])
L = float(info[6])/float(info[2])
l_b = 0.965
f.close()

if 'Film' in sys_typ :
    Lx_Lz_ratio = ''
    if Lx_Lz_ratio == '':
        Lx_Lz_ratio = 1
    Lx_Lz_ratio = float(Lx_Lz_ratio)
    print('film setup',flush=True)
    Lx = L*Lx_Lz_ratio**(1/3)
    Lz = Lx/Lx_Lz_ratio
    
    f = open(path + '/KG/Inputs/{}_KG_0.data'.format(mol))
    for i in range(12):
        line = f.readline().split()
    Lz_str_lo = 2*float(line[0])
    Lz_str_hi = 2*float(line[1])
    f.close()
    
    f_tab_P0 = path + '/KG/Inputs/KG_P0.table'
    

f = open(f_in)
lines = f.readlines()
f.close()

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
        
        lines[l] = 'angle_coeff 1 {}\n'.format(k_prime[0])
    
    if 'LZ_LO_STR' in line and 'Film' in sys_typ :
        lines[l] = 'variable        wall_str_low        index    {}\n'.format(Lz_str_lo)
    if 'LZ_HI_STR' in line and 'Film' in sys_typ :
        lines[l] = 'variable        wall_str_hgh        index    {}\n'.format(Lz_str_hi)
    if 'LZ_LO_END' in line and 'Film' in sys_typ :
        lines[l] = 'variable        wall_end_low        index    0\n'
    if 'LZ_HI_END' in line and 'Film' in sys_typ :
        lines[l] = 'variable        wall_end_hgh        index    {}\n'.format(Lz+2)

if 'Single' in sys_typ or 'Free' in sys_typ:
    pair_table_writer_zero_pressure(f_tab_P0)
    lines.append('\n#===========================================================\n')
    lines.append('# ZERO PRESSURE PAIR POTENTIAL FOR FILM\n')
    lines.append('#===========================================================\n\n')
    lines.append('pair_style table linear 1000\n')
    lines.append('pair_coeff 1 1 {} Nonbonded\nspecial_bonds fene\n\n'.format(f_tab_P0))
    lines.append('run 1000000\n\n')
    lines.append('unfix wall_lo\n')
    lines.append('unfix wall_hi\n\n')
    if 'Single' in sys_typ:
        lines.append('fix wall_lo all wall/lj126 zlo EDGE 1.0 1.0 2.5 units box\n\n')
        lines.append('change_box all z final 0 {} units box\n\n'.format(2*Lz))
    else:
        lines.append('change_box all z final {} {} units box\n\n'.format(-Lz,2*Lz))
    lines.append('run 100000\n\n')
    lines.append('write_data      ${final}')

f = open(f_out.format(mol),'w')
for line in lines :
    f.write(line)
f.close()
