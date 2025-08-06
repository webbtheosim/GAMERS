import numpy as np

from openff.toolkit import ForceField, Molecule, Topology, utils
from openff.interchange import Interchange
from openff.units import unit
from openff.interchange.components._packmol import pack_box

from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

from openmm import XmlSerializer
from openmm.app import Element

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

def read_charge_file(file,N,DoP):
    f = open(file)
    f.readline()
    if DoP == 1:
        charges = np.array([])
        for j in range(N+2):
            charges = np.append(charges,float(f.readline()))
    else :
        charges_f = np.array([])
        charges_m = np.array([])
        charges_b = np.array([])
        for j in range(N+1):
            charges_f = np.append(charges_f,float(f.readline()))
        f.readline()
        for j in range(N):
            charges_m = np.append(charges_m,float(f.readline()))
        f.readline()
        for j in range(N+1):
            charges_b = np.append(charges_b,float(f.readline()))
        f.close()
        
        if DoP == 2:
            charges = np.append(charges_f,charges_b)
        else :
            charges = np.copy(charges_f)
            for i in range(DoP-2):
                charges = np.append(charges,charges_m)
            charges = np.append(charges,charges_b)
        
    return charges*unit.elementary_charge


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

mol = 'PEO'
density = 1.06*unit.gram/unit.centimeter**3
DoP_polymer = 100
N_mol = 30
N_config = 8
connector_idx = -1
path = '/scratch/gpfs/jm7732/Equil/PEO_Paper'
smiles_mon = r'COC'
smiles_head = r''
smiles_tail = r''
charge_file_polymer = '/home/jm7732/Sage_Charge_Catalog/PEO.charges'
idx_config = 0

dr_out = path + '/HR/Inputs'

sage = ForceField("openff_unconstrained-2.2.1.offxml")

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

mass = DoP_polymer*mass_mon*unit.gram/unit.mol + mass_head*unit.gram/unit.mol + mass_tail*unit.gram/unit.mol
L = (N_mol*unit.mol/6.022/10**23*mass/density)**(1/3)
print('box size [nm]:',L.to(unit.nanometer),flush=True)

time_mon = time.time()
print('reading charges',flush=True)

N_head = 1
if smiles_head == 'C':
    N_head = 4

smiles_polymer = smiles_head + smiles_mon*DoP_polymer + smiles_tail
polymer = my_remap(Molecule.from_smiles(smiles_polymer,allow_undefined_stereo=True),DoP_polymer,N_atm_per_mon,N_H_per_hvy,connector_idx,smiles_head,smiles_tail)
polymer.partial_charges = read_charge_file(charge_file_polymer,N_atm_per_mon,DoP_polymer)

if not os.path.exists(dr_out+'/LAMMPS_Backup'):
    os.makedirs(dr_out+'/LAMMPS_Backup')

cubic_box = unit.Quantity(L * np.eye(3), unit.angstrom)

print('generating conformers for trajectory {} with RDKit, more will be generated later if specified in the input file'.format(idx_config),flush=True)
atm_pos = np.zeros((1,3))
for i in range(N_mol):
    print('generating conformer {}'.format(i),flush=True)
    # documentation for EmbedMultipleConfs: https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html
    rdpolymer = polymer.to_rdkit()
    AllChem.EmbedMultipleConfs(rdpolymer, numConfs=1, randomSeed=(i+N_mol*idx_config+23), useRandomCoords=True)
    temp = Molecule.from_rdkit(rdpolymer, allow_undefined_stereo=True)

    for j,conformer in enumerate(temp.conformers) :
        conformer += np.random.uniform(0,1,3)*L
        atm_pos = np.append(atm_pos,conformer,axis=0)
atm_pos = np.delete(atm_pos,0,axis=0)
topology = Topology()
for j in range(N_mol) :
    topology.add_molecule(polymer)
topology.set_positions(atm_pos)

system = Interchange.from_smirnoff(force_field=sage, topology=topology, charge_from_molecules=[polymer], box=cubic_box)

system.to_lammps(dr_out+'/LAMMPS_Backup/{}_{}'.format(mol,idx_config))
system.to_pdb(dr_out+'/{}_openff_formatting_{}.pdb'.format(mol,idx_config))

print('rdkit config generation completed, times are reported below:',flush=True)
print(time.strftime("%Hh%Mm%Ss", time.gmtime(time.time()-time_chg)),'for configuration generation',flush=True)
print('an average of',time.strftime("%Hh%Mm%Ss", time.gmtime((time.time()-time_chg)/N_mol/N_config)),'was required to generate a single chain',flush=True)

