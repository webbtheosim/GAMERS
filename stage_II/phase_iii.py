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
N_steps_anneal = int(float(inputs['annealing_simulation_length(ns)'])*10**6)
N_steps_npt = int(float(inputs['npt_simulation_length(ns)'])*10**6)
path = inputs['output_directory']

pdb_file = path + '/stage_II/inputs/sys.pdb'
system_file = path + '/stage_II/inputs/sys.xml'
ff_file = path + '/stage_II/inputs/ff.xml'
thermo_file = path + '/stage_II/stage_II.thermo'

pos_in_file = path + '/stage_II/final_positions/phase_ii.pos'
vel_in_file = path + '/stage_II/final_positions/phase_ii.vel'

pos_out_file_phase_iii = path + '/stage_II/final_positions/phase_iii.pos'
vel_out_file_phase_iii = path + '/stage_II/final_positions/phase_iii.vel'

pos_out_file_npt = path + '/stage_II/final_positions/npt.pos'
vel_out_file_npt = path + '/stage_II/final_positions/npt.vel'

################################################################################
# Full potential system
################################################################################

print('simulation will be performed at {}'.format(T))
    
pdb = PDBFile(pdb_file)
with open(system_file) as input:
    system = XmlSerializer.deserialize(input.read())
    
integrator = LangevinMiddleIntegrator(T,1/picosecond,0.001*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
with open(pos_in_file) as input:
    input.readline()
    line = input.readline().split()
    box_vec_a = np.array([float(line[0]),float(line[1]),float(line[2])])*nanometers
    line = input.readline().split()
    box_vec_b = np.array([float(line[0]),float(line[1]),float(line[2])])*nanometers
    line = input.readline().split()
    box_vec_c = np.array([float(line[0]),float(line[1]),float(line[2])])*nanometers
    input.readline()
    input_num = int(input.readline())
    pos = np.zeros((input_num,3))
    input.readline()
    for i in range(input_num) :
        line = input.readline().split()
        pos[i] = np.array([float(line[0]),float(line[1]),float(line[2])])*nanometers
with open(vel_in_file) as input:
    input.readline()
    input_num = int(input.readline())
    vel = np.zeros((input_num,3))
    input.readline()
    for i in range(input_num) :
        line = input.readline().split()
        vel[i] = np.array([float(line[0]),float(line[1]),float(line[2])])*nanometer/picosecond
simulation.context.setPeriodicBoxVectors(box_vec_a,box_vec_b,box_vec_c)
simulation.context.setPositions(pos)
simulation.context.setVelocities(vel)
simulation.minimizeEnergy()

simulation.reporters.append(StateDataReporter(stdout, 100000, step=True, potentialEnergy=True, temperature=True, density=True))
safe_reinitialize(simulation)

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
