import numpy as np

from openff.toolkit import ForceField, Molecule, Topology, utils
from openff.interchange import Interchange
from openff.units import unit

from openff.toolkit.utils.toolkits import NAGLToolkitWrapper

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

from openmm import XmlSerializer
from openmm.app import Element

from pathlib import Path

import time
import sys
import os

import decimal
        
def value_to_decimal(value, decimal_places):
    decimal.getcontext().rounding = decimal.ROUND_HALF_UP  # define rounding method
    return decimal.Decimal(str(float(value))).quantize(decimal.Decimal('1e-{}'.format(decimal_places)))

def my_remap(molecule,DoP,atm_per_mon,H_bonded,end_idx,head,tail):
    N_atoms = atm_per_mon*DoP + 2
    
    mapping = {}
    for i in range(N_atoms) :
        mapping[i] = 0
        
    H_bonded_full = np.copy(H_bonded)
    for i in range(DoP-1):
        H_bonded_full = np.append(H_bonded_full,H_bonded)
        
    N_heavy = len(H_bonded)*DoP
    if head == 'C':
        N_atoms += 3
        N_heavy += 1
        H_bonded_full = np.append(3,H_bonded_full)
    else :
        H_bonded_full[0] += 1
    if tail == 'C':
        N_atoms += 3
        N_heavy += 1
        H_bonded_full = np.append(H_bonded_full,3)
    else :
        H_bonded_full[end_idx] += 1
        
    idx_heavy = 0
    idx_atom = 0
    idx_H = 0
    end_atom_idx = 0
    
    for i in range(N_heavy) :
        mapping[idx_heavy] = idx_atom
        idx_heavy += 1
        idx_atom += 1
        
        if i == 0 or i == N_heavy + end_idx :
            end_atom_idx = idx_atom-1
        
        for j in range(H_bonded_full[i]) :
            mapping[N_heavy+idx_H] = idx_atom
            idx_H += 1
            idx_atom += 1
    
    mapping_H1 = {0:1,1:0}
    for i in range(2,len(mapping)) :
        mapping_H1[i] = i
    if end_idx != -1 :
        mapping_H1[end_atom_idx+1] = len(mapping)-1
        for i in range(end_atom_idx+2,len(mapping)) :
            mapping_H1[i] = i-1
    molecule = molecule.remap(mapping)
    molecule = molecule.remap(mapping_H1)
    
    return molecule
    
def tpi_file_maker(file,N_mol,N_mon,N_atm_per_mon,head,tail):
    if head == 'C':
        N_head = 4
    else :
        N_head = 1
    if tail == 'C':
        N_tail = 4
    else :
        N_tail = 1
    
    f = open(file,'w')
    f.write('# each entry is for an atom, with the order being the same as the .xml file\n# molecule_id monomer_id')
    for i in range(N_mol):
        for j in range(N_head):
            f.write('\n{} {}'.format(i+1,1))
        for j in range(N_mon):
            for k in range(N_atm_per_mon):
                f.write('\n{} {}'.format(i+1,j+1))
        for j in range(N_tail):
            f.write('\n{} {}'.format(i+1,N_mon))
    f.close()

def read_charge_file(file,N,DoP):
    import numpy as np

def round_and_renormalize_charges(charges, decimals=8):
    """
    Round partial charges to a fixed number of decimals and renormalize
    so they sum to exactly zero.

    Parameters
    ----------
    charges : array-like
        Input array of partial charges (list, tuple, or numpy array).
    decimals : int, optional
        Number of decimal places to round to (default = 6).

    Returns
    -------
    np.ndarray
        Array of rounded and renormalized charges.
    """
    charges = np.asarray(charges.m_as(unit.elementary_charge), dtype=float)

    correction = charges.sum()
    charges -= correction/len(charges)
    charges = np.round(charges, decimals)
    correction = charges.sum()
    counts = int(correction*10**decimals)
    for count in range(abs(counts)):
        if correction > 0:
            charges[count] -= 10**(-decimals)
        elif correction < 0:
            charges[count] += 10**(-decimals)
    charges = np.round(charges, decimals)
    
    return charges*unit.elementary_charge

def config_generator(pos_mon,DoP,idx_connector):
    pos_mon = pos_mon.m_as(unit.nanometer)
    pos_mon_extender = np.copy(pos_mon[1:])-pos_mon[1]
    vec0 = (pos_mon[idx_connector]-pos_mon[0])/np.linalg.norm(pos_mon[idx_connector]-pos_mon[0])
    theta = np.arccos(vec0[2])
    phi = np.sign(vec0[1])*np.arccos(vec0[0]/np.sqrt(vec0[0]**2+vec0[1]**2))
    pos_pol = np.copy(pos_mon)
    idx_monomers = np.ones(len(pos_mon)-1,dtype=int)
    for i in range(DoP-1):
        phi += np.random.uniform(-np.pi/4,np.pi/4)
        theta += np.random.uniform(-np.pi/4,np.pi/4)
        vec = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        R = R_b_to_a(vec,vec0)
        pos_temp = np.copy(pos_mon_extender)
        for j in range(len(pos_mon_extender)):
            pos_temp[j] = np.matmul(R,pos_temp[j])
        pos_pol = np.append(pos_pol[:-1],pos_temp+pos_pol[-1],axis=0)
        idx_monomers = np.append(idx_monomers,np.ones(len(pos_mon)-2,dtype=int)*(i+2))
    idx_monomers = np.append(idx_monomers,DoP)
    return pos_pol*unit.nanometer, idx_monomers
    
def R_b_to_a(a,b):
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    R = np.eye(3)+vx+np.matmul(vx,vx)*1/(1+c)
    return R

def pdb_formatter(mol,idx_monomers,f_in,f_out):

    if len(mol) > 3:
        mol = mol[-3:]

    idx_mol = 1
    idx_atm = 0
    idx_atm_tot = 1
    
    dict_idx = {}

    lines = []
    f = open(f_in)
    lines.append(f.readline())
    line = f.readline()
    while 'CONECT' not in line :
        if 'TER' in line :
            idx_mol += 1
            idx_atm = 0
        else :
            dict_idx[line[6:11]] = idx_atm_tot
            line = 'ATOM  ' + '{:5d}'.format(idx_atm_tot) + ' {:04X}'.format(idx_atm) + ' {:3} {}{:4}'.format(mol,line[21],idx_monomers[idx_atm]) + line[26:]
            idx_atm += 1
            idx_atm_tot += 1
            lines.append(line)
        line = f.readline()
    while line != '' :
        line_new = 'CONECT'
        for i in range(4):
            if len(line) > 8 + 5*i:
                line_new = line_new + '{:5d}'.format(dict_idx[line[6+5*i:11+5*i]])
        if len(line_new) > 10:
            lines.append(line_new + '\n')
        else:
            lines.append(line)
        line = f.readline()
    f.close()
    
    f = open(f_out,'w')
    for line in lines :
        f.write(line)
    f.close()

################################################################################
#
# Goal: produce files ready for OpenMM (and LAMMPS) simulation with am1bcc
#       charges even for long polymers
#
# Key steps for polymers:
#       generate oligomer (odd DoP, <100 atoms) from smiles string
#       generate and save am1bcc oligomer charges
#       generate polymer from smiles string
#       apply end monomer charges of oligomer to polymer end monomers
#       apply middle monomer charge of oligomer to remaining polymer monomers
#       output to OpenMM (save xml) and LAMMPS data file
#
################################################################################

start = time.time()

with open(Path(__file__).resolve().parent / 'GAMERS.input') as input:
    lines = input.readlines()
    inputs = {}
    for line in lines[:-1]:
        inputs[line.split(':')[0]] = line.strip().split(':')[1]

mol = inputs['polymer_name']
density = float(inputs['density(g/cm^3)'])*unit.gram/unit.centimeter**3
DoP_polymer = int(inputs['degree_of_polymerization'])
N_mol = int(inputs['molecule_count'])
connector_idx = int(inputs['smiles_connection_index'])
smiles_mon = inputs['monomer_smiles']
smiles_head = inputs['head_smiles']
smiles_tail = inputs['tail_smiles']
path = inputs['output_directory']

dr_out = path + '/stage_II/inputs'

sage = ForceField('openff_unconstrained-2.2.1.offxml')

# set molecule values
try:
    monomer = Molecule.from_smiles(smiles_mon,allow_undefined_stereo=True)
except:
    monomer = Molecule.from_smiles(smiles_mon+'([H])',allow_undefined_stereo=True)
N_atm_per_mon = -2
N_hvy_per_mon = 0
if smiles_head == 'C':
    mass_head = Element.getBySymbol('C').mass + 3*Element.getBySymbol('H').mass
else:
    mass_head = Element.getBySymbol('H').mass
if smiles_tail == 'C':
    mass_tail = Element.getBySymbol('C').mass + 3*Element.getBySymbol('H').mass
else:
    mass_tail = Element.getBySymbol('H').mass
mass_mon = -2*Element.getBySymbol('H').mass
for s in monomer.to_smiles() :
    if s.isalpha() :
        N_atm_per_mon += 1
        mass_mon += Element.getBySymbol(s).mass
        if s != 'H' :
            N_hvy_per_mon += 1
N_H_per_hvy = np.zeros(N_hvy_per_mon, dtype=int)
N_H_per_hvy[0] -= 1
N_H_per_hvy[connector_idx] -= 1
hvy_idx = 0
for atom in monomer.atoms :
    if atom.atomic_number != 1 :
        for bond in monomer.bonds:
            if bond.atom1 == atom or bond.atom2 == atom:
                # Check if the other atom in the bond is a hydrogen
                other_atom = bond.atom2 if bond.atom1 == atom else bond.atom1
                if other_atom.atomic_number == 1:  # Atomic number 1 is Hydrogen
                    N_H_per_hvy[hvy_idx] += 1
        hvy_idx +=1
print('monomer SMILES:',smiles_mon,flush=True)
print('number of hydrogens per heavy atom:',N_H_per_hvy,flush=True)

monomer = my_remap(monomer,1,N_atm_per_mon,N_H_per_hvy,connector_idx,'','')
monomer.to_file(dr_out + '/mon.pdb', file_format='pdb')

time_mon = time.time()

smiles_polymer = smiles_head + smiles_mon*DoP_polymer + smiles_tail
polymer = my_remap(Molecule.from_smiles(smiles_polymer,allow_undefined_stereo=True),DoP_polymer,N_atm_per_mon,N_H_per_hvy,connector_idx,smiles_head,smiles_tail)

NAGLToolkitWrapper().assign_partial_charges(polymer, 'openff-gnn-am1bcc-1.0.0.pt')

polymer.partial_charges = round_and_renormalize_charges(polymer.partial_charges, decimals=6)

if not os.path.exists(dr_out+'/LAMMPS_Backup'):
    os.makedirs(dr_out+'/LAMMPS_Backup')

mass = DoP_polymer*mass_mon*unit.gram/unit.mol + mass_head*unit.gram/unit.mol + mass_tail*unit.gram/unit.mol
L = (N_mol*unit.mol/6.022/10**23*mass/density)**(1/3)
box_dim = np.array([[L.m_as(unit.nanometer),0,0],[0,L.m_as(unit.nanometer),0],[0,0,L.m_as(unit.nanometer)]])*unit.nanometer

print('generating a single conformers with RDKit, more will be generated later if specified in the input file',flush=True)

rdmonomer = monomer.to_rdkit()
# documentation for EmbedMultipleConfs: https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html
AllChem.EmbedMultipleConfs(rdmonomer, numConfs=1, randomSeed=(23), useRandomCoords=True)
temp = Molecule.from_rdkit(rdmonomer, allow_undefined_stereo=True)
    
for i,conformer in enumerate(temp.conformers) :
    monomer_pos = conformer

time_config = time.time()

print('saving system to .pdb and LAMMPS data file',flush=True)

polymer_pos, idx_mon = config_generator(monomer_pos,DoP_polymer,connector_idx)
atm_pos = np.zeros((1,3))
for i in range(N_mol):
    atm_pos = np.append(atm_pos,polymer_pos + np.random.uniform(0,1,3)*L,axis=0)
atm_pos = np.delete(atm_pos,0,axis=0)
topology = Topology()
for j in range(N_mol) :
    topology.add_molecule(polymer)
topology.set_positions(atm_pos)

system = Interchange.from_smirnoff(force_field=sage, topology=topology, charge_from_molecules=[polymer], box=box_dim)

system.to_lammps(dr_out+'/LAMMPS_Backup/sys')
system.to_pdb(dr_out+'/sys_openff_formatting.pdb')


pdb_formatter(mol,idx_mon,dr_out+'/sys_openff_formatting.pdb',dr_out+'/sys.pdb')

openmm_sys = system.to_openmm()

with open(dr_out + '/sys.xml', 'w') as output:
    output.write(XmlSerializer.serialize(openmm_sys))

tpi_file_maker(dr_out + '/sys.tpi',N_mol,DoP_polymer,N_atm_per_mon,smiles_head,smiles_tail)

print('openff setup completed, times are reported below:',flush=True)
print(time.strftime("%Hh%Mm%Ss", time.gmtime(time_mon-start)),'for general setup',flush=True)
print(time.strftime("%Hh%Mm%Ss", time.gmtime(time_config-time_mon)),'for configuration generation',flush=True)
print(time.strftime("%Hh%Mm%Ss", time.gmtime(time.time()-time_config)),'for topology file generation',flush=True)

