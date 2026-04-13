from openff.toolkit import Molecule
import numpy as np
import sys
from scipy.interpolate import InterpolatedUnivariateSpline

from pathlib import Path


def edge_padding(crds) :
    
    d_s = crds[1] - crds[0]
    d_e = crds[-2] - crds[-1]
    
    start = np.array([crds[0]-3*d_s,crds[0]-2*d_s,crds[0]-d_s])
    end = np.array([crds[-1]-d_e,crds[-1]-2*d_e,crds[-1]-3*d_e])
    
    return np.append( start, np.append(crds, end, axis=0), axis=0 )

def hydrogen_tracker(connector_idx,smiles_mon,smiles_head,smiles_tail):
    ### NEW ADDITION: treats F as H, as both are minimal for restraint CoM ###
    ### calculation, which are memory intensive, so reducing atoms is best ###
    try:
        monomer = Molecule.from_smiles(smiles_mon,allow_undefined_stereo=True)
    except:
        monomer = Molecule.from_smiles(smiles_mon+'([H])',allow_undefined_stereo=True)
    N_hvy_per_mon = 0
    connector_shift = 0
    for s in monomer.to_smiles() :
        if s.isalpha() :
            if s not in ['H','F'] :
                N_hvy_per_mon += 1
                connector_shift = 0
            elif s == 'F':
                connector_shift += 1
    connector_idx += connector_shift
    N_H_per_hvy = np.zeros(N_hvy_per_mon, dtype=int)
    N_H_per_hvy[0] -= 1
    N_H_per_hvy[connector_idx] -= 1
    hvy_idx = 0
    for atom in monomer.atoms :
        if atom.atomic_number not in [1,9] :
            for bond in monomer.bonds:
                if bond.atom1 == atom or bond.atom2 == atom:
                    # Check if the other atom in the bond is a hydrogen
                    other_atom = bond.atom2 if bond.atom1 == atom else bond.atom1
                    if other_atom.atomic_number == 1 or other_atom.atomic_number == 9:
                        N_H_per_hvy[hvy_idx] += 1
            hvy_idx +=1
    
    if smiles_head == 'C':
        N_atm_head = 4
    else:
        N_atm_head = 1
    if smiles_tail == 'C':
        N_atm_tail = 4
    else:
        N_atm_tail = 1
    
    return N_atm_head, N_atm_tail, N_H_per_hvy

def chunk_file_maker(file,N_mol,N_tether,N_tether_higher,mon_per_tether,connector_idx,smiles_mon,smiles_head,smiles_tail) :

    N_atm_head, N_atm_tail, N_H_per_hvy = hydrogen_tracker(connector_idx,smiles_mon,smiles_head,smiles_tail)
        
    f = open(file,'w')
    atm_idx = 0
    for i in range(N_mol):
        atm_idx += N_atm_head
        for j in range(N_tether):
            if j < N_tether_higher:
                temp_mon_per_tether = mon_per_tether + 1
            else:
                temp_mon_per_tether = mon_per_tether
            for k in range(temp_mon_per_tether):
                for idx, N_H in enumerate(N_H_per_hvy):
                    f.write('{} {}\n'.format(atm_idx+1,j+i*N_tether+1))
                    atm_idx += 1 + N_H
        atm_idx += N_atm_tail
    f.close()

with open(Path(__file__).resolve().parent.parent / 'GAMERS.input') as input:
    lines = input.readlines()
    inputs = {}
    for line in lines[:-1]:
        inputs[line.split(':')[0]] = line.strip().split(':')[1]

N_mon = int(inputs['degree_of_polymerization'])
connector_idx = int(inputs['smiles_connection_index'])
smiles_mon = inputs['monomer_smiles']
smiles_head = inputs['head_smiles']
smiles_tail = inputs['tail_smiles']
path = inputs['output_directory']

f_KG_in = path + '/stage_I/step_2.end'
f_mol = path + '/stage_I/step_3'

f = open(path+'/stage_I/step_1.mapping')
f.readline()
N_b, kappa, sigma, epsilon, M_b, tau, L, T, N_tether = f.readline().split()
f.close()
N_b = int(N_b)
k = float(kappa)
sigma = float(sigma)
L = float(L)/10 #converted to nanometers
N_tether = int(N_tether)

mon_per_tether = N_mon//N_tether
N_tether_higher = N_mon%N_tether
ratio_t_to_b = N_tether/N_b

f = open(f_KG_in,'r')
for i in range(2) :
    f.readline()
N = int(f.readline().split()[0])
N_mol = int(N/N_b)
line = f.readline()
while 'xlo' not in line :
    line = f.readline()
L_KG = float(f.readline().split()[1])
scale = L/L_KG
while 'Atoms' not in line :
    line = f.readline()
f.readline()
crds_KG = np.zeros( (N,3) )
crds_tether = np.zeros( (N_mol,N_tether,3) )
images = np.zeros( (N_mol,N_b,3) )
for i in range(N) :
    line = f.readline().split()
    i = int(line[0])-1
    x = float(line[3]) + int(line[6])*L_KG
    y = float(line[4]) + int(line[7])*L_KG
    z = float(line[5]) + int(line[8])*L_KG
    crds_KG[i] = np.array([x,y,z])
    images[i//N_b][i%N_b] = crds_KG[i]//L_KG
f.close()

crds_KG *= scale
crds_tether *= scale

r = float(ratio_t_to_b)
edge = 0.5
N_avg = 100
ts = np.linspace(-edge,N_b-1+edge,N_b*N_avg)
# creating list of indexes that map ts to tethers
ts_to_tethers = [0]
sum_mon = 0
for i in range(N_tether):
    if i < N_tether_higher:
        sum_mon += mon_per_tether + 1
    else:
        sum_mon += mon_per_tether
    ts_to_tethers.append(int(N_avg*sum_mon*N_b/N_mon))
#print(ts_to_tethers,len(ts))

for i in range(N_mol) :
    img_avg = np.mean(images[i],axis=0)
    crds_KG[i*N_b:(i+1)*N_b] -= L*np.array([round(img_avg[0]),round(img_avg[1]),round(img_avg[2])])

f = open(f_mol + '_restraint.pos','w')
f.write('{}\n# tether positions in nanometers\n'.format(N_mol*N_tether))
for i in range(N_mol) :
    crds_temp = edge_padding(crds_KG[i*N_b:(i+1)*N_b])
    spline_x = InterpolatedUnivariateSpline(np.arange(-3,N_b+3),crds_temp[:,0])
    spline_y = InterpolatedUnivariateSpline(np.arange(-3,N_b+3),crds_temp[:,1])
    spline_z = InterpolatedUnivariateSpline(np.arange(-3,N_b+3),crds_temp[:,2])
    x = spline_x(ts)
    y = spline_y(ts)
    z = spline_z(ts)
    for j in range(N_tether):
        mon_idx0 = ts_to_tethers[j]
        mon_idx1 = ts_to_tethers[j+1]
        f.write('B {} {} {}\n'.format(np.mean(x[mon_idx0:mon_idx1]),np.mean(y[mon_idx0:mon_idx1]),np.mean(z[mon_idx0:mon_idx1])))
f.close()

chunk_file_maker(f_mol+'_restraint.ids',N_mol,N_tether,N_tether_higher,mon_per_tether,connector_idx,smiles_mon,smiles_head,smiles_tail)

print('Process complete', flush=True)
