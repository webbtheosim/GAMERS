from openmm.app import *
from openmm import *
from openmm.unit import *
from openff.toolkit import Molecule
import numpy as np
from sys import stdout, argv

from scipy.optimize import fsolve
from functools import partial
from numba import jit

import time
import sys

def theta_from_kappa(k,l_b):
    return 2*np.arctan( np.sqrt( l_b / kuhn_from_kappa(k,l_b) ) )

def kuhn_from_kappa(k, l_b):
    if k == 0 :
        return l_b + 0.77*(np.tanh(-0.03*k**2 - 0.41*k + 0.16) + 1)
    else :
        return l_b*( 2*k + np.exp(-2*k) -1 )/( 1 - np.exp(-2*k) * (2*k + 1) ) + 0.77*(np.tanh(-0.03*k**2 - 0.41*k + 0.16) + 1)

def k_solve(k_prime,k,W):
    
    l_b = 0.965
    l_k0 = l_b*(2*k+np.exp(-2*k)-1)/(1-np.exp(-2*k)*(2*k+1))
    dl_k = 0.7276*(1+np.tanh(0.002*k**3-0.469*k+0.214))
    l_k0W = l_b*(2*k_prime+np.exp(-2*k_prime)-1)/(1-np.exp(-2*k_prime)*(2*k_prime+1))
    dl_kW = 0.3638*(1-np.tanh(-5.481*W**3-3.870*W+2.318))*(1+np.tanh(0.002*k_prime**3-0.469*k_prime+0.214))
    
    return l_k0 + dl_k - l_k0W - dl_kW
    
def next_point_random_rotation(v,theta) :
    v_norm = v/np.linalg.norm(v)
    t_1 = np.cross(v,np.array([-v_norm[2],v_norm[0],v_norm[1]]))
    t_2 = np.cross(v_norm,t_1)
    angle = np.random.uniform(0,2*np.pi)
    rot_vec = t_1/np.linalg.norm(t_1) * np.sin(angle) + t_2/np.linalg.norm(t_2) * np.cos(angle)
    return v * np.cos(theta) + np.cross(rot_vec,v)*np.sin(theta) + rot_vec * np.dot(rot_vec,v) * (1 - np.cos(theta))

def combine_dims(a, start=0, count=2):
    """ Reshapes numpy array a by combining count dimensions, 
        starting at dimension index start """
    s = a.shape
    return np.reshape(a, s[:start] + (-1,) + s[start+count:])

@jit
def windowCount(crds, center, R, L):
    dx = np.abs(crds[:,0] - center[0])
    dy = np.abs(crds[:,1] - center[1])
    dz = np.abs(crds[:,2] - center[2])
    
    count = 0
    for i in range(len(dx)) :
        if dx[i] > L/2 :
            dx[i] = (L - dx[i])
        if dy[i] > L/2 :
            dy[i] = (L - dy[i])
        if dz[i] > L/2 :
            dz[i] = (L - dz[i])
        if np.sqrt( dx[i]**2 + dy[i]**2 + dz[i]**2 ) < R :
            count += 1
    return count

def frc(kappa,N_mon,N_mol,sys_typ):
    
    N = N_mon*N_mol
    L = (N/0.85)**(1/3)
    if 'Film' in sys_typ :
        Lx_Lz_ratio = ''
        if Lx_Lz_ratio == '':
            Lx_Lz_ratio = 1
        Lx_Lz_ratio = float(Lx_Lz_ratio)
        print('film setup',flush=True)
        L = L*Lx_Lz_ratio**(1/3)
    else :
        print('bulk setup',flush=True)
    
    N_MC = 1000
    N_var = 10
    R = 5
    
    l_b = 0.965
    theta = theta_from_kappa(kappa,l_b)
    
    crds = np.zeros( (N_mol,N_mon,3) )
    R_2 = np.zeros(N_mol)
    R_2_thr = l_b**2 * (1 + np.cos(theta))/(1 - np.cos(theta))
    
    ##############################################################################
    # Chain Creation
    ##############################################################################
    
    print('\nGenerating chains via the freely-rotating-chain algorithm\n', flush=True)
    i = 0
    while i < N_mol :
        # the first position is fixed at (0,0,0), the second is placed using  uniformly
        # sampled theta and phi, then the chain is built and randomly shifted 
        theta_0 = np.random.uniform(0,np.pi)
        phi_0 = np.random.uniform(0,2*np.pi)
        crds[i][1] = l_b * np.array([ np.sin(theta_0)*np.sin(phi_0), np.sin(theta_0)*np.cos(phi_0), np.cos(theta_0) ])
        
        # the nested loop is over the number of beads per chain, starting at the 3rd bead
        for j in range(2,N_mon) :
            crds[i][j] = crds[i][j-1] + next_point_random_rotation((crds[i][j-1]-crds[i][j-2]),theta)
        crds[i] += np.random.uniform(0,L,3)
        R_2[i] = np.sum(np.square(crds[i][-1] - crds[i][0]))
            
        i += 1
        if i == N_mol :
            if (0.9*R_2_thr > np.mean(R_2)/N_mon or np.mean(R_2)/N_mon > 1.1*R_2_thr) and N_mon > 20 :
                i = 0
                crds = np.zeros((N_mol,N_mon,3))
                print('regenerating confiruation, R_2 of {:.2f} is not close enough to theoretical value of {:.2f}'.format(np.mean(R_2)/N_mon,R_2_thr), flush=True)
            else :
                print('completed confiruation, R_2 of {:.2f} is close to theoretical value of {:.2f}'.format(np.mean(R_2)/N_mon,R_2_thr), flush=True)
    print('\nR_2: simulation {:.2f} theory {:.2f}'.format(np.mean(R_2)/N_mon, R_2_thr), flush=True)
    print('l_k: simulation {:.2f} theory {:.2f}'.format(np.mean(R_2)/N_mon/l_b, kuhn_from_kappa(kappa, l_b)), flush=True)
    
    ##############################################################################
    # Zero-Temp Monte Carlo
    ##############################################################################
    
    print('\nPerforming zero-temp MC to reduce fluctuations', flush=True)
    Ns = np.zeros(N_var)
    for i in range(N_var) :
        Ns[i] = windowCount(combine_dims(crds)%L, np.random.uniform(0,L,3), R, L)
    var = np.var(Ns)
    print('\ninitial number variance: {:.2f}'.format(var))
    for i in range(N_MC) :
        idx = np.random.randint(N_mol)
        shift = np.random.uniform(0,L,3)
        crds_temp = np.copy(crds)
        crds_temp[idx] += shift
        for j in range(N_var) :
            Ns[j] = windowCount(combine_dims(crds_temp)%L, np.random.uniform(0,L,3), R, L)
        var_temp = np.var(Ns)
        if var_temp < var :
            var = var_temp
            crds[idx] += shift
            print('number variance reduced to {:.2f} on MC attempt {}'.format(var,i), flush=True)
    
    ##############################################################################
    # Moving each molecule into the box
    ##############################################################################
    
    print('\nMoving chains into the box\n', flush=True)
    for i in range(N_mol) :
        image = np.zeros((N_mon, 3))
        for l in range(N_mon) :
            image[l] = np.floor(crds[i][l]/L)
        image_avg = np.mean(image,axis=0)
        crds[i] -= L*np.array([round(image_avg[0]),round(image_avg[1]),round(image_avg[2])])
    
    return crds,L
    
def kuhn_mapping(N_mon,N_mol,T,rho,M_mon,M_k,l_k,smiles,min_mon_per_tether):
    
    # we want the average monomer mass, so we get the mass of a chain and divide by N_mon
    try:
        monomer = Molecule.from_smiles(smiles,allow_undefined_stereo=True)
    except:
        monomer = Molecule.from_smiles(smiles+'([H])',allow_undefined_stereo=True)
    M_openff = -2*Element.getBySymbol('H').mass.in_units_of(gram/mole)
    for s in monomer.to_smiles() :
        if s.isalpha() :
            M_openff += Element.getBySymbol(s).mass.in_units_of(gram/mole)
    
    N_A = 6.02214076*10**23/mole
    k_B = 8.31446262/10**3*kilojoule/mole/kelvin
    rho_b, l_b = (0.85,0.965)
    
    M_k = M_k*M_openff/M_mon
    M_mon = M_openff
    
    print('l_k = {}\nM_k = {}'.format(l_k,M_k),flush=True)
    
    rho_k = rho/M_k*N_A
    n_k = rho_k*l_k**3
    N_k_per_mon = M_mon/M_k
    c_b = np.sqrt(n_k/rho_b/l_b**3)
    N_b_per_mon = c_b*N_k_per_mon
    N_b = N_b_per_mon*N_mon
    
    func = lambda k : rho_b*l_b*(l_b*(2*k+np.exp(-2*k)-1)/(1-np.exp(-2*k)*(2*k+1))+0.77*(np.tanh(-0.03*k**2-0.41*k+0.16)+1))**2-n_k 
    kappa = fsolve(func, 1)[0]
    sigma = (l_k/l_b/c_b).in_units_of(nanometers)
    epsilon = k_B*T
    M_b = M_k/c_b
    
    # we need an integer number of beads, and thus must scale various parameters
    # slightly so the resulting melt is still the correct size
    sigma = sigma*(N_b/round(N_b))**(1/3)
    M_b = M_b*N_b/round(N_b)
    N_b = round(N_b)
    
    tau = (sigma*sqrt(M_b/epsilon)).in_units_of(picosecond)
    L = (N_b*N_mol/0.85)**(1/3)*sigma
    
    mon_per_tether = max(int(np.ceil(M_k/M_mon)),min_mon_per_tether)
    #while N_mon%mon_per_tether != 0 :
    #    mon_per_tether += 1
    N_tether = N_mon//mon_per_tether
    
    return N_b, kappa, sigma, epsilon, M_b, tau, L, T, N_tether

mol = 'PEO'
rho = 1.06*gram/centimeter**3
T = 353*kelvin
min_mon_per_tether = 0
l_k = 0.971*nanometer
M_k = 117.12*gram/mole
M_mon = 44.03*gram/mole
N_mon = 100
N_mol = 30
N_trj = 8
smiles = r'COC'
path = '/scratch/gpfs/jm7732/Equil/PEO_Paper'
sys_typ = ''

f_map = path + '/KG/{}_KG.mapping'.format(mol)
f_out = path + '/KG/Inputs/{}_KG'.format(mol) + '_{}.data'

print('Performing kuhn mapping', flush=True)
N_b, kappa, sigma, epsilon, M_b, tau, L, T, N_tether = kuhn_mapping(N_mon,N_mol,T,rho,M_mon,M_k,l_k,smiles,min_mon_per_tether)

print('N_b = {}\nkappa = {}\nsigma = {}\nepsilon = {}\nM_b = {}\ntau = {}\nL = {}\nT = {}\nN_tether = {}'.format(N_b, kappa, sigma, epsilon, M_b, tau, L, T, N_tether),flush=True)

f = open(f_map,'w')
f.write('# N_b, kappa, sigma (A), epsilon (kcal/mol), M_b (g/mol), tau (fs), L (A), T (Kelvin), N_tether\n')
f.write('{} {} {} {} {} {} {} {} {}'.format(N_b, kappa, sigma.value_in_unit(angstrom), epsilon.value_in_unit(kilocalorie/mole), M_b.value_in_unit(gram/mole), tau.value_in_unit(femtosecond), L.value_in_unit(angstrom), T.value_in_unit(kelvin), N_tether))
f.close()

N = N_b*N_mol

for idx in range(N_trj) :

    print('Calculating initial positions for trajectory',idx, flush=True)
    positions,L = frc(kappa,N_b,N_mol,sys_typ)
    
    ##############################################################################
    # Writing Data File
    ##############################################################################
    
    print('\nWriting data file')
    
    N_bond = N - N_mol
    N_angle = N - 2*N_mol

    f = open(f_out.format(idx), 'w')
    f.write('#LAMMPS data file generated with python script of Jacob Metcalfe to simulate freely-rotating chain\n\n')
    f.write('{} atoms\n'.format( int(N_b*N_mol) ))
    f.write('1 atom types\n')
    f.write('{} bonds\n'.format(N_bond))
    f.write('1 bond types\n')
    f.write('{} angles\n'.format(N_angle))
    f.write('1 angle types\n')
    f.write('\n0 {} xlo xhi\n'.format(L))
    f.write('0 {} ylo yhi\n'.format(L))
    if 'Film' in sys_typ:
        f.write('{} {} zlo zhi\n\n'.format(2*np.amin(positions[:,:,2]),2*np.amax(positions[:,:,2])))
    else:
        f.write('0 {} zlo zhi\n\n'.format(L))
    f.write('Masses\n\n1 1\n\n')
    f.write('Atoms # angle\n\n')
    
    for i in range(N_mol) :
        for j in range(N_b) :
            if 'Film' in sys_typ:
                f.write('{} {} 1 {} {} {} {} {} 0\n'.format(i*N_b+j+1,i+1,positions[i][j][0]%L,positions[i][j][1]%L,positions[i][j][2],int(positions[i][j][0]//L),int(positions[i][j][1]//L)))
            else:
                f.write('{} {} 1 {} {} {} {} {} {}\n'.format(i*N_b+j+1,i+1,positions[i][j][0]%L,positions[i][j][1]%L,positions[i][j][2]%L,int(positions[i][j][0]//L),int(positions[i][j][1]//L),int(positions[i][j][2]//L)))
    f.write('\nBonds\n\n')
    c = 0
    for i in range(N_mol) :
        for j in range(N_b-1) :
            c += 1
            k = i*N_b + j + 1
            f.write('{} 1 {} {}\n'.format(c,k,k+1))
    f.write('\nAngles\n\n')
    c = 0
    for i in range(N_mol) :
        for j in range(N_b-2) :
            c += 1
            k = i*N_b + j + 1
            f.write('{} 1 {} {} {}\n'.format(c,k,k+1,k+2))
    f.close()
