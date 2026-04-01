import numpy as np
import openmm
from openmm import app, unit, XmlSerializer

from datetime import datetime


from tqdm import tqdm

import argparse

# %% Argument parser

path_pdb = './top.pdb'
path_sys = './system.xml'

steps = 40000000        # number of steps (timestep = 10 fs)
wfreq = 10000         # trajectory saving frequency
logfreq = 10000       # logfile frequency

random_number_seed = None

fcheck_out = './restart.chk'

# %% 

parser = argparse.ArgumentParser(description="Run molecular dynamics simulation")
parser.add_argument(
    "--nthreads",
    type=int,
    default=1,
    help="Number of threads to use (default: 1)"
)
args = parser.parse_args()

threads = args.nthreads

temp = 260  # loaded system is specific to temperature

pdb = app.pdbfile.PDBFile(path_pdb)

with open(path_sys, 'r') as infile:
    xml_content = infile.read()
    system = XmlSerializer.deserialize(xml_content)

# use langevin integrator
integrator = openmm.openmm.LangevinMiddleIntegrator(temp*unit.kelvin,0.01/unit.picosecond,0.01*unit.picosecond)
if random_number_seed is not None:
    integrator.setRandomNumberSeed(random_number_seed)
print(integrator.getFriction(),integrator.getTemperature())

platform = openmm.Platform.getPlatformByName('CPU')
simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform, dict(Threads=str(threads)))

simulation.context.setPositions(pdb.positions)

# Emin
print('Minimizing energy.')
simulation.minimizeEnergy()

# run simulation
simulation.reporters.append(app.dcdreporter.DCDReporter('./traj.dcd',wfreq,append=False, enforcePeriodicBox=True))
simulation.reporters.append(app.statedatareporter.StateDataReporter('./sim.log',logfreq,
        step=True,speed=True,elapsedTime=True,potentialEnergy=False,separator='\t',append=False))

print("STARTING SIMULATION", flush=True)
nbatches = 10
batch = int(steps / nbatches)
for i in tqdm(range(nbatches),mininterval=1):
    simulation.step(batch)
    simulation.saveCheckpoint(fcheck_out)
simulation.saveCheckpoint(fcheck_out)

now = datetime.now()
# dt_string = now.strftime("%Y%d%m_%Hh%Mm%Ss")

state_final = simulation.context.getState(getPositions=True,enforcePeriodicBox=True)
rep = app.pdbreporter.PDBReporter(f'./final_conf.pdb',0)
rep.report(simulation,state_final)
rep = app.pdbreporter.PDBReporter(f'./checkpoint.pdb',0)
rep.report(simulation,state_final)




