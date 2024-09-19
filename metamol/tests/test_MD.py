import pytest
import os
from pkg_resources import resource_filename
from distutils.spawn import find_executable

import metamol as meta
from metamol.lib.molecules.alkane import *
from metamol.lib.molecules.polymer import *
from metamol.lib.fragments.monomers import *
from metamol.utils.help_functions import save_remove

def polymer_system(monomers, seq, head, tail, N, sol):
    pol = Polymer(monomers, seq=seq, head=head, tail=tail)
    pol.build(N)
    pol.embed()
    pol_sys = meta.System([pol, sol], [60//N, 100], box=(60, 60, 60), name='POLSYS')
    ff_file = resource_filename("metamol", os.path.join("ff_files", "opls.xml"))
    pol_sys.parametrize(forcefield_files=ff_file)
    pol_sys.initial_config()
    return pol_sys

def run_lammps(pol_sys, sol, gpu, gpu_backend):
    from metamol.engines.metaLammps import metaLammps

    data_file = resource_filename("metamol", os.path.join("tests", "files", 'temp.data'))
    pol_sys.save(data_file)

    work_dir = resource_filename("metamol", os.path.join("tests", "files"))
    numTypes = len(pol_sys.params['atom_type'])
    wtype1, wtype2 = numTypes-1, numTypes
    numBTypes = len(pol_sys.params['bond_type'])
    numATypes = len(pol_sys.params['angle_type'])

    mlmp = metaLammps()

    # Pass in LAMMPS commands
    mlmp.command('clear')
    mlmp.command('units real')
    mlmp.command('dimension 3')
    mlmp.command('boundary p p p')
    mlmp.command('atom_style full')

    mlmp.command('pair_style lj/long/coul/long long long 20.0')
    mlmp.command('pair_modify mix geometric')
    mlmp.command('pair_modify table 0')
    mlmp.command('pair_modify table/disp 0')

    mlmp.command('bond_style harmonic')
    mlmp.command('angle_style harmonic')
    mlmp.command('dihedral_style opls')

    mlmp.command('read_data temp.data')
    if 'Water' in sol.name:
        mlmp.command('group water type {0:d} {1:d}'.format(wtype1, wtype2))
        mlmp.command('group PEG type 1:{0:d}'.format(wtype1-1))

    mlmp.command('velocity all create 300.0 4928459 dist gaussian')
    mlmp.command('kspace_style ewald/disp 5.0e-6')

    mlmp.command('neighbor 2.0 bin')
    mlmp.command('neigh_modify every 1 delay 0 check yes')

    if 'Water' in sol.name:
        mlmp.command('fix 1 water shake 1.0e-7 100 100 b {0:d} a {1:d}'.format(numBTypes, numATypes))
    mlmp.command('thermo 10')
    mlmp.command('thermo_style custom step temp press vol density pe ke evdwl ecoul epair ebond eangle edihed eimp emol elong etotal atoms')
    mlmp.command('thermo_modify lost error norm no')
    mlmp.command('timestep 1.0')
    mlmp.command('fix 2 all nvt temp 300.0 300.0 100.0')

    mlmp.command('run 1000')
    mlmp.command('write_data temp_out.data')

    mlmp.replace("run 1000", "run 100")
    mlmp.launch(work_dir=work_dir, output='out.dat', mpi=False, gpu=gpu, gpu_backend=gpu_backend)

    # Analyze thermo output from simulations
    temp_out = mlmp.get_thermo('Temp')

    pe_out = mlmp.get_thermo('PotEng')

    # Close LAMMPS
    mlmp.close()

    save_remove(data_file)
    save_remove(os.path.join(work_dir, 'out.dat'))
    save_remove(os.path.join(work_dir, 'temp_out.data'))
    save_remove(os.path.join(work_dir, 'log.lammps'))
    save_remove('log.lammps')

def run_gromacs(pol_sys, gpu):
    from metamol.engines.metaGromacs import metaGromacs

    gro_file = resource_filename("metamol", os.path.join("tests", "files", 'temp.gro'))
    top_file = resource_filename("metamol", os.path.join("tests", "files", 'temp.top'))
    pol_sys.save(gro_file)
    pol_sys.save(top_file)

    work_dir = resource_filename("metamol", os.path.join("tests", "files"))

    mgro = metaGromacs()

    # Pass in Gromacs commands
    mgro.command('title = dppc')
    mgro.command('cpp = /lib/cpp')
    mgro.command('integrator = md')
    mgro.command('nsteps = 1000')
    mgro.command('nstlist = 10')

    mgro.commands_list(['nstfout = 0', 'nstxout = 0', 'nstvout = 0', 'nstxtcout = 0', 'nstlog = 0'])

    mgro.command('dt = 0.001')
    mgro.command('constraints = all-bonds')
    mgro.command('nstenergy = 0')
    mgro.command('ns_type = grid')
    mgro.command('coulombtype = PME')

    mgro.commands_list(['rlist = 2.0', 'rvdw = 2.0', 'rcoulomb = 2.0', 'tcoupl = v-rescale', 'tc_grps = system'])
    mgro.command('tau_t = 0.1')
    mgro.command('ref_t = 300')
    mgro.command('fourier_spacing = 0.125')
    mgro.command('nstcalcenergy = 100')
    mgro.command('cutoff-scheme = verlet')

    # Run Gromacs
    mgro.grompp(work_dir=work_dir, gro_file=gro_file, top_file=top_file)
    extra_cmds = '-nb gpu -bonded gpu -pme gpu' if gpu else ''
    mgro.mdrun(work_dir=work_dir, extra_cmds=extra_cmds)

    # Close Gromacs instance
    mgro.close()

    save_remove(gro_file)
    save_remove(top_file)
    save_remove(os.path.join(work_dir, 'md.log'))
    save_remove(os.path.join(work_dir, 'mdout.mdp'))
    save_remove(os.path.join(work_dir, 'sys.mdp'))
    save_remove(os.path.join(work_dir, 'ener.edr'))
    save_remove(os.path.join(work_dir, 'topol.tpr'))

def run_openmm(topo, pos, system, platform):
    from sys import stdout
    import openmm as mm

    # Create the integrator to do Langevin dynamics
    integrator = mm.LangevinIntegrator(
                            300*mm.unit.kelvin,       # Temperature of heat bath
                            1.0/mm.unit.picoseconds,  # Friction coefficient
                            1.0*mm.unit.femtoseconds, # Time step
    )
    # Create the Simulation object
    simulation = mm.app.Simulation(topo, system, integrator, platform)

    # Set the particle positions
    simulation.context.setPositions(pos)

    # Minimize the energy
    simulation.minimizeEnergy()

    # Report eneries every 1,000 steps
    simulation.reporters.append(mm.app.PDBReporter('output.pdb', 1000))
    simulation.reporters.append(
            mm.app.StateDataReporter(stdout, 100, step=True, potentialEnergy=True,
                                kineticEnergy=True, temperature=True)
    )

    # Run MD simulations
    simulation.step(5000)

    return

def __get_param__(monomers_seq=[], head=[], tail=[], N=[], sol=[]):
    for m_seq in monomers_seq:
        for h in head:
            for t in tail:
                for num in N:
                    for s in sol:
                        yield (m_seq[0], m_seq[1], h, t, num, s)

LMP = find_executable("lmp_serial") or find_executable("lmp_mpi") or find_executable("lmp")
@pytest.mark.skipif(
    not LMP, reason="Lammps executable not installed or found"
)
@pytest.mark.parametrize('monomers,seq,head,tail,N,sol', __get_param__(
    monomers_seq = [([CH2(), PEGMonomer()], 'AB')],
    head = [None, OH(), CH3()], tail = [None, CH3()],
    N = [5, 10],
    sol = [Ethane(), meta.SPCE()],
    )
)
def test_lammps(monomers, seq, head, tail, N, sol, gpu, gpu_backend):
    pol_sys = polymer_system(monomers, seq, head, tail, N, sol)
    run_lammps(pol_sys, sol, gpu, gpu_backend)

GMX = find_executable("gmx") or find_executable("gmx_mpi")
@pytest.mark.skipif(
    not GMX, reason="Gromacs executable not installed or found"
)
@pytest.mark.parametrize('monomers,seq,head,tail,N,sol', __get_param__(
    monomers_seq = [([CH2(), PEGMonomer()], 'AB')],
    head = [None, OH(), CH3()], tail = [None, CH3()],
    N = [5, 10],
    sol = [Ethane(), meta.SPCE()],
    )
)
def test_gromacs(monomers, seq, head, tail, N, sol, gpu, gpu_backend):
    pol_sys = polymer_system(monomers, seq, head, tail, N, sol)
    run_gromacs(pol_sys, gpu)

@pytest.mark.parametrize('monomers,seq,head,tail,N,sol', __get_param__(
    monomers_seq = [([CH2(), PEGMonomer()], 'AB')],
    head = [OH(), CH3()], tail = [CH3()],
    N = [5, 10],
    sol = [Ethane(), meta.SPCE()],
    )
)
def test_openmm(monomers, seq, head, tail, N, sol):
    import openmm as mm

    #print(os.environ)
    pol_sys = polymer_system(monomers, seq, head, tail, N, sol)
    topo, pos, system = pol_sys.to_openmm(createSystem=True, constraints=mm.app.HBonds)

    numPlatforms = mm.Platform.getNumPlatforms()
    for idx in range(1, numPlatforms):
        platform = mm.Platform.getPlatform(idx)
        if platform.getName() == "OpenCL": continue
        print("Testing platform: ", platform.getName())

        run_openmm(topo=topo, pos=pos, system=system, platform=platform)
        # Read the output system and compare to the original one
        sys_out = meta.System('output.pdb')

        assert sys_out.numAtoms == pol_sys.numAtoms
        assert sys_out.numBonds == pol_sys.numBonds
        assert sys_out.numMols == pol_sys.numMols

        save_remove('ouput.pdb')

    return
