import numpy as np

from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from sys import argv

from copy import deepcopy


def tpi_load(file,N):
    f = open(file)
    lines = f.readlines()
    output = np.zeros((len(lines)-2,2),dtype=int)
    for i,l in enumerate(lines[2:]):
        l = l.split()
        output[i] = np.array([int(l[0]),int(l[1])])
    f.close()
    if N != len(output):
        raise ValueError('The provided .xml and .tpi files have differing atom counts')
    return output

def pull_forcefield_generator_simple(system,scale14_LJ,scale14_coul) :
    forces = {system.getForce(index).getName(): system.getForce(index) for index in range(system.getNumForces())}
    if 'LennardJones' in [f.getName() for f in system.getForces()]:
      original_nonbonded_force = deepcopy(forces['LennardJones'])
    else:
      original_nonbonded_force = deepcopy(forces['Nonbonded force'])
    LJ = CustomNonbondedForce('4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=(sigma1+sigma2)/2; epsilon=sqrt(epsilon1*epsilon2); chargeprod=charge1*charge2')
    energy_expression = 'lambda_p^2*(4*epsilon*(1/(0.5*(1-lambda_p)^2+(r/sigma)^6)^2-1/(0.5*(1-lambda_p)^2+(r/sigma)^6)) + 138.9354576*chargeprod/(0.1^2*(1-lambda_p)^2+r^2)^0.5); sigma=(sigma1+sigma2)/2; epsilon=sqrt(epsilon1*epsilon2); chargeprod=charge1*charge2'
    LJ = CustomNonbondedForce(energy_expression)
    LJ.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    LJ.addPerParticleParameter('sigma')
    LJ.addPerParticleParameter('epsilon')
    LJ.addPerParticleParameter('charge')
    LJ.setCutoffDistance(original_nonbonded_force.getCutoffDistance())
    LJ.setUseLongRangeCorrection(False)
    LJ.addGlobalParameter('lambda_p',0)

    for index in range(original_nonbonded_force.getNumParticles()):
      charge, sigma, epsilon = original_nonbonded_force.getParticleParameters(index)
      LJ.addParticle([sigma, epsilon, charge])
    
    energy_expression = 'lambda_p^2*({}*4*epsilon*(1/(0.5*(1-lambda_p)^2+(r/sigma)^6)^2-1/(0.5*(1-lambda_p)^2+(r/sigma)^6)) + {}*138.9354576*chargeprod/(0.1*(1-lambda_p)^2+r^2)^0.5)'
    LJ_14 = CustomBondForce(energy_expression.format(scale14_LJ,scale14_coul))
    LJ_14.addPerBondParameter('sigma')
    LJ_14.addPerBondParameter('epsilon')
    LJ_14.addPerBondParameter('chargeprod')
    LJ_14.addGlobalParameter('lambda_p',0)

    for index in range(original_nonbonded_force.getNumExceptions()):
      j, k, chargeprod, sigma, epsilon = original_nonbonded_force.getExceptionParameters(index)
      LJ_14.addBond(j, k, [sigma, epsilon, chargeprod])
      LJ.addExclusion(j, k)

    LJ.setName('LJ_soft')
    LJ_14.setName('LJ_soft_14')
    
    for i,f in reversed(list(enumerate(system.getForces()))) :
        if f.getName() in ['Nonbonded force','LennardJones'] :
            system.removeForce(i)

    return system, LJ, LJ_14, original_nonbonded_force

def tether_centroid(system,simulation,groups_file,crds_file,force_coeff,force_type) :
    f = open(groups_file)
    lines = f.readlines()
    f.close()
    groups = []
    for line in lines :
        atom_idx = int(line.split()[0])
        group_idx = int(line.split()[1])
        if len(groups) < group_idx :
            groups.append([])
        groups[group_idx-1].append(atom_idx-1)
    
    # the crds file contains 4 values per line: the name (irrelivant) and xyz crds
    f = open(crds_file)
    lines = f.readlines()
    f.close()
    crds = np.zeros((len(lines),3))
    for i,line in enumerate(lines[2:]) :
        crds[i] = np.array([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])*nanometer
    
    # determine the force per tether if the based on mass of system
    mass_system = 0*amu
    for molecule in simulation.context.getMolecules() :
        for particle_idx in molecule :
            mass_system += system.getParticleMass(particle_idx)
    mass_group = mass_system/len(groups)
    force = mass_group * force_coeff*kilojoule/mole/amu/nanometer**2
    
    group_val = 0
    group_count = 0
    force_count = 0
    
    if force_type == 'capped' :
        tether = CustomCentroidBondForce(1, 'lambda_r*k*w*sqrt((x1-x)^2+(y1-y)^2+(z1-z)^2)*tanh(sqrt((x1-x)^2+(y1-y)^2+(z1-z)^2)/w)')
        tether.addGlobalParameter('w',0.1*nanometer)
    if force_type == 'harmonic' :
        tether = CustomCentroidBondForce(1, 'lambda_r*k*((x1-x)^2+(y1-y)^2+(z1-z)^2)')
    tether.addGlobalParameter('lambda_r',1)
    tether.addGlobalParameter('k',force)
    tether.addPerBondParameter('x')
    tether.addPerBondParameter('y')
    tether.addPerBondParameter('z')
    tether.setUsesPeriodicBoundaryConditions(False)
    tether.setName('tether_{}'.format(force_count))
    
    while group_count < len(groups) :
        tether.addGroup(groups[group_count])
        tether.addBond([group_val],crds[group_count])
        
        group_count += 1
        
        if group_count == len(groups) :
            system.addForce(tether)
        elif group_val == 31 :
            system.addForce(tether)
            force_count += 1
            if force_type == 'capped' :
                tether = CustomCentroidBondForce(1, 'lambda_r*k*w*sqrt((x1-x)^2+(y1-y)^2+(z1-z)^2)*tanh(sqrt((x1-x)^2+(y1-y)^2+(z1-z)^2)/w)')
                tether.addGlobalParameter('w',0.1*nanometer)
            if force_type == 'harmonic' :
                tether = CustomCentroidBondForce(1, 'lambda_r*k*((x1-x)^2+(y1-y)^2+(z1-z)^2)')
            tether.addGlobalParameter('lambda_r',1)
            tether.addGlobalParameter('k',force)
            tether.addPerBondParameter('x')
            tether.addPerBondParameter('y')
            tether.addPerBondParameter('z')
            tether.setUsesPeriodicBoundaryConditions(False)
            tether.setName('tether_{}'.format(force_count))
            group_val = 0
        else :
            group_val += 1
            
    return system

def save_pos_vel_files(state,pos_out_file,vel_out_file):
    box = state.getPeriodicBoxVectors()/nanometers
    N_atm = len(state.getPositions())
    with open(pos_out_file, 'w') as output:
        output.write('Box vectors #reported in nm\n')
        output.write('{} {} {}\n{} {} {}\n{} {} {}\n'.format(box[0][0],box[0][1],box[0][2],box[1][0],box[1][1],box[1][2],box[2][0],box[2][1],box[2][2]))
        output.write('Atom entries\n{}\nAtom positions #reported in nm\n'.format(N_atm))
        for pos in state.getPositions()/nanometers :
            output.write('{} {} {}\n'.format(pos[0],pos[1],pos[2]))
    with open(vel_out_file, 'w') as output:
        output.write('Atom entries\n{}\nAtom velocities #reported in nm per ps\n'.format(N_atm))
        for vel in state.getVelocities()/nanometer*picosecond :
            output.write('{} {} {}\n'.format(vel[0],vel[1],vel[2]))

class pos_saver(object):
    def __init__(self, reportInterval, file_pos, file_vel):
        self._reportInterval = reportInterval
        self._file_pos = file_pos[:-4] + '_{}ns' + '.pos'
        self._file_vel = file_vel[:-4] + '_{}ns' + '.vel'

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return {'steps': steps, 'periodic': None, 'include':['positions', 'velocities']}

    def report(self, simulation, state):
        time = round(simulation.currentStep/10**6)
        save_pos_vel_files(simulation.context.getState(getPositions=True,getVelocities=True),self._file_pos.format(time),self._file_vel.format(time))

################################################################################
# Parameters
################################################################################

T = 353*kelvin
sys_typ = ''
pdb_file = '/scratch/gpfs/jm7732/Equil/PEO_Paper/HR/Inputs/PEO_0.pdb'
system_file = '/scratch/gpfs/jm7732/Equil/PEO_Paper/HR/Inputs/PEO.xml'
thermo_file = '/scratch/gpfs/jm7732/Equil/PEO_Paper/HR/Outputs/PEO_0.thermo'
groups_file = '/scratch/gpfs/jm7732/Equil/PEO_Paper/KG/Outputs/PEO_tether.ids'
crds_file = '/scratch/gpfs/jm7732/Equil/PEO_Paper/KG/Outputs/PEO_tether_0.pos'
tether_prefactor = 1
N_steps_pull = 1*10**6
N_steps_rest = 10**5
N_steps_anneal = 4*10**6
N_steps_npt = 200*10**6
pos_out_file_pull = '/scratch/gpfs/jm7732/Equil/PEO_Paper/HR/Final_Positions/PEO_pull_0.pos'
pos_out_file_anneal = '/scratch/gpfs/jm7732/Equil/PEO_Paper/HR/Final_Positions/PEO_anneal_0.pos'
pos_out_file_npt = '/scratch/gpfs/jm7732/Equil/PEO_Paper/HR/Final_Positions/PEO_npt_0.pos'
vel_out_file_pull = '/scratch/gpfs/jm7732/Equil/PEO_Paper/HR/Final_Positions/PEO_pull_0.vel'
vel_out_file_anneal = '/scratch/gpfs/jm7732/Equil/PEO_Paper/HR/Final_Positions/PEO_anneal_0.vel'
vel_out_file_npt = '/scratch/gpfs/jm7732/Equil/PEO_Paper/HR/Final_Positions/PEO_npt_0.vel'

################################################################################
# Monomer tethers with no pair potential
################################################################################

print('simulation will be performed at {}'.format(T))
    
pdb = PDBFile(pdb_file)
with open(system_file) as input:
    system = XmlSerializer.deserialize(input.read())

integrator = LangevinMiddleIntegrator(T,1/picosecond,0.001*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
system, LJ_soft, LJ_soft_14, original_nonbonded_force = pull_forcefield_generator_simple(system,0.5,0.833333)
simulation.context.reinitialize(preserveState=True)

simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

if tether_prefactor > 0:
    system = tether_centroid(system,simulation,groups_file,crds_file,tether_prefactor,'capped')
simulation.context.reinitialize(preserveState=True)
        
simulation.reporters.append(StateDataReporter(stdout, 100000, step=True, potentialEnergy=True, temperature=True, density=True))

for f in system.getForces():
    if 'tether' not in f.getName() :
        print(f.getName())
    else :
        t_w = simulation.context.getParameter('w')

if tether_prefactor > 0:
    print('Pulling chains to KG contour')
    # tether width is increased over 100 steps
    #print('Switching to harmonic spring')
    for i in range(100) :
        simulation.context.setParameter('w',(i+1)**2/100*t_w)
        simulation.step(N_steps_pull//100)
    
    for i,f in enumerate(system.getForces()):
        if 'tether' in f.getName():
            f.setEnergyFunction('lambda_r*k*((x1-x)^2+(y1-y)^2+(z1-z)^2)')

system.addForce(LJ_soft)
system.addForce(LJ_soft_14)

simulation.context.reinitialize(preserveState=True)

print('Introducing pair potentials')
lambdas = np.delete(np.linspace(0,1,101),0)
# 100 intervals to introduce lambda_p
for l in lambdas:
    simulation.context.setParameter('lambda_p',l)
    simulation.minimizeEnergy()

remove = []
for i,f in enumerate(system.getForces()) :
    if 'LJ' in f.getName():
        remove.append(i)
for i in reversed(remove):
    system.removeForce(i)
system.addForce(original_nonbonded_force)
simulation.context.reinitialize(preserveState=True)
simulation.minimizeEnergy()

for f in system.getForces() :
    if 'tether' not in f.getName() :
        print(f.getName())

if tether_prefactor > 0:
    print('Removing restraints')
    lambdas = np.delete(np.linspace(1,0,101),0)
    # 100 intervals to remove lambda_r
    for l in lambdas:
        simulation.context.setParameter('lambda_r',l)
        simulation.step(N_steps_rest//100)
    
    remove = []
    for i,f in enumerate(system.getForces()) :
        if 'tether' in f.getName():
            remove.append(i)
    for i in reversed(remove):
        system.removeForce(i)

save_pos_vel_files(simulation.context.getState(getPositions=True,getVelocities=True),pos_out_file_pull,vel_out_file_pull)

################################################################################
# Annealing of the system to/from 1.5x T
################################################################################

if N_steps_anneal != 0:

    Th = 1.5*T
    system.addForce(CMMotionRemover(1000))
    simulation.context.reinitialize(preserveState=True)
    
    print('Ramping T={} to T={}'.format(T,Th))
    for i in range(100) :
        T_tmp = T+(Th-T)*(i+1)/100
        integrator.setTemperature(T_tmp)
        simulation.step(N_steps_anneal//100//4)
        
    print('Holding at T={}'.format(Th))
    integrator.setTemperature(Th)
    simulation.step(N_steps_anneal//4)
    
    print('Ramping T={} to T={}'.format(Th,T))
    for i in range(100) :
        T_tmp = Th+(T-Th)*(i+1)/100
        integrator.setTemperature(T_tmp)
        simulation.step(N_steps_anneal//100//4)
        
    print('Holding at T={}'.format(T))
    integrator.setTemperature(T)
    simulation.step(N_steps_anneal//4)
    
    save_pos_vel_files(simulation.context.getState(getPositions=True,getVelocities=True),pos_out_file_anneal,vel_out_file_anneal)

################################################################################
# NPT equilibration
################################################################################

if N_steps_npt != 0:
    simulation.reporters.append(StateDataReporter(thermo_file, 1000, step=True, density=True))
    simulation.reporters.append(pos_saver(10**7,pos_out_file_npt,vel_out_file_npt))
    system.addForce(MonteCarloBarostat(1*bar, T))
    simulation.context.reinitialize(preserveState=True)
    
    print("Running NPT")
    simulation.step(N_steps_npt)
    
    save_pos_vel_files(simulation.context.getState(getPositions=True,getVelocities=True),pos_out_file_npt,vel_out_file_npt)
