import numpy as np

from openmm import *
from openmm.app import *
from openmm.unit import *
from sys import stdout
from sys import argv

from copy import deepcopy

from pathlib import Path


class Geometric(object):
  def __init__(self,system,scale14):
    self._system = system
    self._scale14 = scale14

  def GeometricMix(system,scale14):
    forces = {system.getForce(index).__class__.__name__: system.getForce(
      index) for index in range(system.getNumForces())}
    nonbonded_force = forces['NonbondedForce']
    geo = CustomNonbondedForce(
      '4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)')
    geo.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    geo.addPerParticleParameter('sigma')
    geo.addPerParticleParameter('epsilon')
    geo.setCutoffDistance(nonbonded_force.getCutoffDistance())
    geo.setUseLongRangeCorrection(True)
    system.addForce(geo)
    LJset = {}
    for index in range(nonbonded_force.getNumParticles()):
      charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
      LJset[index] = (sigma, epsilon)
      geo.addParticle([sigma, epsilon])
      nonbonded_force.setParticleParameters(
        index, charge, sigma, epsilon * 0)
    for i in range(nonbonded_force.getNumExceptions()):
      (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
      geo.addExclusion(p1, p2)
      if eps._value != 0.0:
        sig14 = (LJset[p1][0] * LJset[p2][0])**0.5
        eps14 = scale14*(LJset[p1][1] * LJset[p2][1])**0.5 #DEPENDING ON FORCE FIELD, MODIFY THIS
        nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps14)
    return system

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

def pull_forcefield_generator_custom_force_present(system,scale14_LJ,scale14_coul) :
    forces = {system.getForce(index).getName(): system.getForce(index) for index in range(system.getNumForces())}
    
    original_nonbonded_force = deepcopy(forces['NonbondedForce'])
    original_nonbonded_custom = deepcopy(forces['CustomNonbondedForce'])
    
    LJ = CustomNonbondedForce('4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2); chargeprod=charge1*charge2')
    energy_expression = 'lambda_p^2*(4*epsilon*(1/(0.5*(1-lambda_p)^2+(r/sigma)^6)^2-1/(0.5*(1-lambda_p)^2+(r/sigma)^6)) + 138.9354576*chargeprod/(0.1^2*(1-lambda_p)^2+r^2)^0.5); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2); chargeprod=charge1*charge2'
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
      sigma, epsilon = original_nonbonded_custom.getParticleParameters(index)
      if sigma == 0:
        sigma = 0.1
      LJ.addParticle([sigma, epsilon, charge])

    energy_expression = 'lambda_p^2*({}*4*epsilon*(1/(0.5*(1-lambda_p)^2+(r/sigma)^6)^2-1/(0.5*(1-lambda_p)^2+(r/sigma)^6)) + {}*138.9354576*chargeprod/(0.1*(1-lambda_p)^2+r^2)^0.5)'
    LJ_14 = CustomBondForce(energy_expression.format(scale14_LJ,scale14_coul))
    LJ_14.addPerBondParameter('sigma')
    LJ_14.addPerBondParameter('epsilon')
    LJ_14.addPerBondParameter('chargeprod')
    LJ_14.addGlobalParameter('lambda_p',0)

    for index in range(original_nonbonded_force.getNumExceptions()):
      j, k, chargeprod, sigma, epsilon = original_nonbonded_force.getExceptionParameters(index)
      if sigma == 0:
        sigma = 0.1
      LJ_14.addBond(j, k, [sigma, epsilon, chargeprod])
      LJ.addExclusion(j, k)

    LJ.setName('LJ_soft')
    LJ_14.setName('LJ_soft_14')
    
    for i,f in reversed(list(enumerate(system.getForces()))) :
        if f.getName() in ['Nonbonded force','NonbondedForce','CustomNonbondedForce','LennardJones','CMMotionRemover'] :
            system.removeForce(i)

    return system, LJ, LJ_14, original_nonbonded_force, original_nonbonded_custom

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

def safe_reinitialize(simulation):
    state = simulation.context.getState(getPositions=True,getVelocities=True)
    positions = state.getPositions()
    velocities = state.getVelocities()
    box = state.getPeriodicBoxVectors()
    simulation.context.reinitialize(preserveState=False)
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(velocities)
    simulation.context.setPeriodicBoxVectors(*box)

################################################################################
# Parameters
################################################################################

with open(Path(__file__).resolve().parent.parent / 'GAMERS.input') as input:
    lines = input.readlines()
    inputs = {}
    for line in lines[:-1]:
        inputs[line.split(':')[0]] = line.strip().split(':')[1]

T = float(inputs['temperature(K)'])*kelvin
tether_prefactor = float(inputs['restrained_force_prefactor(kJ/mol/amu/nm^2)'])
N_steps_pull = int(float(inputs['restrained_simulation_length(ns)'])*10**6)
N_steps_rest = 10**5
N_steps_anneal = int(float(inputs['annealing_simulation_length(ns)'])*10**6)
N_steps_npt = int(float(inputs['npt_simulation_length(ns)'])*10**6)
path = inputs['output_directory']

pdb_file = path + '/stage_II/inputs/sys.pdb'
system_file = path + '/stage_II/inputs/sys.xml'
ff_file = path + '/stage_II/inputs/ff.xml'
thermo_file = path + '/stage_II/stage_II.thermo'
groups_file = path + '/stage_I/step_3_restraint.ids'
crds_file = path + '/stage_I/step_3_restraint.pos'

pos_out_file_phase_i = path + '/stage_II/final_positions/phase_i.pos'
pos_out_file_phase_ii = path + '/stage_II/final_positions/phase_ii.pos'
pos_out_file_phase_iii = path + '/stage_II/final_positions/phase_iii.pos'
pos_out_file_npt = path + '/stage_II/final_positions/npt.pos'
vel_out_file_phase_i = path + '/stage_II/final_positions/phase_i.vel'
vel_out_file_phase_ii = path + '/stage_II/final_positions/phase_ii.vel'
vel_out_file_phase_iii = path + '/stage_II/final_positions/phase_iii.vel'
vel_out_file_npt = path + '/stage_II/final_positions/npt.vel'

################################################################################
# Kuhn restraints with no pair potential
################################################################################

print('simulation will be performed at {}'.format(T))
    
pdb = PDBFile(pdb_file)
with open(system_file) as input:
    system = XmlSerializer.deserialize(input.read())
    
system, LJ_soft, LJ_soft_14, original_nonbonded_force = pull_forcefield_generator_simple(system,0.5,0.833333)
#system, LJ_soft, LJ_soft_14, original_nonbonded_force, original_nonbonded_custom = pull_forcefield_generator_custom_force_present(system,0.5,0.5)

integrator = LangevinMiddleIntegrator(T,1/picosecond,0.001*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

if tether_prefactor > 0:
    system = tether_centroid(system,simulation,groups_file,crds_file,tether_prefactor,'capped')
safe_reinitialize(simulation)
        
simulation.reporters.append(StateDataReporter(stdout, 100000, step=True, potentialEnergy=True, temperature=True, density=True))

for f in system.getForces():
    if 'tether' not in f.getName() :
        print(f.getName())
    else :
        t_w = simulation.context.getParameter('w')

if tether_prefactor > 0:
    print('Pulling chains to KG contour')
    # tether width is increased over 100 steps
    # print('Switching to harmonic spring')
    for i in range(100) :
        simulation.context.setParameter('w',(i+1)**2/100*t_w)
        simulation.step(N_steps_pull//100)
    
    for i,f in enumerate(system.getForces()):
        if 'tether' in f.getName():
            f.setEnergyFunction('lambda_r*k*((x1-x)^2+(y1-y)^2+(z1-z)^2)')

    save_pos_vel_files(simulation.context.getState(getPositions=True,getVelocities=True),pos_out_file_phase_i,vel_out_file_phase_i)

system.addForce(LJ_soft)
system.addForce(LJ_soft_14)
safe_reinitialize(simulation)

################################################################################
# Introduce pair interactions and remove restraints (concurrent for stability)
################################################################################

if tether_prefactor > 0:
    print('Introducing pair potentials and removing restraints')
    lambdas = np.delete(np.linspace(0,1,901),0)
    # 900 intervals to introduce lambda_p and reduce lambda_r
    for l in lambdas:
        # lambda_pair from 0 to 1
        simulation.context.setParameter('lambda_p',l)
        # lambda_restraint from 1 to 0.1
        simulation.context.setParameter('lambda_r',(1-0.9*l))
        simulation.step(9*N_steps_rest//1000)
    print('Pair potentials fully introduced')
    remove = []
    for i,f in enumerate(system.getForces()) :
        if 'LJ' in f.getName():
            remove.append(i)
    for i in reversed(remove):
        system.removeForce(i)
    system.addForce(original_nonbonded_force)
    #system.addForce(original_nonbonded_custom)
    safe_reinitialize(simulation)

    for f in system.getForces() :
        if 'tether' not in f.getName() :
            print(f.getName())

    lambdas = np.delete(np.linspace(0.1,0,101),0)
    # 100 intervals to finish removing lambda_r
    for l in lambdas:
        # lambda restraint from 0.1 to 0
        simulation.step(N_steps_rest//1000)
    
    remove = []
    for i,f in enumerate(system.getForces()) :
        if 'tether' in f.getName():
            remove.append(i)
    for i in reversed(remove):
        system.removeForce(i)
    
    print('Restarints fully removed')
else:
    print('Introducing pair potentials)
    lambdas = np.delete(np.linspace(0,1,901),0)
    # 100 intervals to introduce lambda_p
    for l in lambdas:
        # lambda pair from 0 to 1
        simulation.context.setParameter('lambda_p',l)
        simulation.step(N_steps_rest//100)
    print('Pair potentials fully introduced')
    remove = []
    for i,f in enumerate(system.getForces()) :
        if 'LJ' in f.getName():
            remove.append(i)
    for i in reversed(remove):
        system.removeForce(i)
    system.addForce(original_nonbonded_force)
    #system.addForce(original_nonbonded_custom)

safe_reinitialize(simulation)
save_pos_vel_files(simulation.context.getState(getPositions=True,getVelocities=True),pos_out_file_phase_ii,vel_out_file_phase_ii)

################################################################################
# Annealing of the system to/from 1.5x T
################################################################################

if N_steps_anneal != 0:

    Th = 1.5*T
    system.addForce(CMMotionRemover(1000))
    safe_reinitialize(simulation)
    
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
    
    save_pos_vel_files(simulation.context.getState(getPositions=True,getVelocities=True),pos_out_file_phase_iii,vel_out_file_phase_iii)

################################################################################
# NPT equilibration
################################################################################

if N_steps_npt != 0:
    simulation.reporters.append(StateDataReporter(thermo_file, 1000, step=True, density=True))
    system.addForce(MonteCarloBarostat(1*bar, T))
    safe_reinitialize(simulation)
    
    print("Running NPT")
    simulation.step(N_steps_npt)
    
    save_pos_vel_files(simulation.context.getState(getPositions=True,getVelocities=True),pos_out_file_npt,vel_out_file_npt)
