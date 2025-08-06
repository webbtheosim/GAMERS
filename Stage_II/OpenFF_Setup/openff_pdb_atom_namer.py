################################################################################
#
# This file replaces the atom names in the current pdb file from openff with
# unique atom names, specifally the atom index in a molecule in hexidecimal, 
# and correctly labels molecule index using 'TER' as an indicator
#
################################################################################

mol = 'PEO'
N_trj = 8
path = '/scratch/gpfs/jm7732/Equil/PEO_Paper'

f_in = path + '/HR/Inputs/{}_openff_formatting'.format(mol) + '_{}.pdb'
f_out = path + '/HR/Inputs/{}'.format(mol) + '_{}.pdb'

if len(mol) > 3:
    mol = mol[:3]

for i in range(N_trj):
    idx_mol = 1
    idx_atm = 0
    
    lines = []
    f = open(f_in.format(i))
    lines.append(f.readline())
    line = f.readline()
    while 'CONECT' not in line :
        if 'TER' in line :
            line = line[:12] + '     {:3}   {:3}\n'.format(mol,idx_mol)
            idx_mol += 1
            idx_atm = 0
        else :
            line = line[:12] + '{:04X}'.format(idx_atm) + ' {:3}{:6}'.format(mol,idx_mol) + line[26:]
            idx_atm += 1
        lines.append(line)
        line = f.readline()
    while line != '' :
        lines.append(line)
        line = f.readline()
    f.close()
    
    f = open(f_out.format(i),'w')
    for line in lines :
        f.write(line)
    f.close()