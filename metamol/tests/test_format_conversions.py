import pytest
import os
from pkg_resources import resource_filename
import numpy as np
from distutils.spawn import find_executable

import metamol as meta
from metamol.utils.help_functions import save_remove

#TODO: gromacs to lammps, openmmm, etc.
@pytest.mark.parametrize('filename', 
['peg100.mol2', 'fullerene.pdb', 'cholesterol.pdb', 'Villin.pdb'])
def test_pmd(filename):
    name, ext_in = filename.split('.')
    file_in = resource_filename("metamol", os.path.join("tests", "files", filename))
    sys = meta.System(file_in, name=name)
    # Parametrize the System
    try:
        import openmm as mm
        sys.parametrize(backend='openmm', forcefield_files=['amber14-all.xml', 'amber14/tip3pfb.xml'], nonbondedMethod=mm.app.PME,
            nonbondedCutoff=1*mm.unit.nanometer, constraints=mm.app.HBonds)
    except:
        sys.parametrize()
    struct_from_sys = sys.to_pmd(parametrize=True)

    assert sys.numBonds == len(struct_from_sys.bonds)
    assert sys.numAngles == len(struct_from_sys.angles)
    assert sys.numDihedrals == len(struct_from_sys.dihedrals)
    assert sys.numRBs == len(struct_from_sys.rb_torsions)
    assert sys.numImpropers == len(struct_from_sys.impropers)

    from metamol.utils.help_functions import parmed_load
    struct_in = parmed_load(file_in)

    assert len(struct_from_sys.atoms) == len(struct_in.atoms)
    assert len(struct_from_sys.residues) == len(struct_in.residues)
    this_atoms = sorted([a.atomic_number for a in struct_from_sys.atoms])
    other_atoms = sorted([a.atomic_number for a in struct_in.atoms])
    assert this_atoms == other_atoms
    assert np.allclose(struct_from_sys.coordinates, struct_in.coordinates, atol=1.0e-3)

    return

GMX = find_executable("gmx") or find_executable("gmx_mpi")
@pytest.mark.skipif(
    not GMX, reason="Gromacs package not installed"
)
@pytest.mark.parametrize('filename', ['Villin.pdb', #['dhfr_pme.top', 'dhfr_pme.gro'],
                                        ['dhfr_gas.top', 'dhfr_gas.gro']]
)
def test_openmm_from_file(filename):
    import openmm as mm
    from sys import stdout

    sys_in = meta.System()
    need_parametrize = False
    gas_sys = False
    if isinstance(filename, str): filename = [filename]
    for f in filename:
        if f.endswith('pdb'): need_parametrize = True
        if 'gas' in f: gas_sys = True
        input_file = resource_filename("metamol", os.path.join("tests", "files", f))
        sys_in.readfile(input_file)
    ff_files = ['amber14-all.xml', 'amber14/tip3pfb.xml'] if need_parametrize else None
    if gas_sys:
        topo, pos, system = sys_in.to_openmm(createSystem=True, 
                                            nonbondedMethod=mm.app.NoCutoff,
                                            constraints=mm.app.HBonds,
                                            implicitSolvent=mm.app.GBn2, 
                                            implicitSolventSaltConc=0.1*mm.unit.moles/mm.unit.liter
        )
    else:
        topo, pos, system = sys_in.to_openmm(createSystem=True, 
                                            forcefield_files=ff_files, 
                                            nonbondedMethod=mm.app.PME,
                                            nonbondedCutoff=8*mm.unit.angstroms, 
                                            constraints=mm.app.HBonds
        )
    
    integrator = mm.LangevinMiddleIntegrator(300*mm.unit.kelvin, 
                                            1/mm.unit.picosecond, 
                                            1.0*mm.unit.femtoseconds)

    # Set platform
    platform = mm.Platform.getPlatformByName('Reference')

    simulation = mm.app.Simulation(topo, system, integrator, platform)
    simulation.context.setPositions(pos)
    if not gas_sys:
        simulation.minimizeEnergy(maxIterations=100)
    simulation.reporters.append(mm.app.PDBReporter('output.pdb', 50))
    simulation.reporters.append(mm.app.StateDataReporter(stdout, 10, step=True,
            potentialEnergy=True, kineticEnergy=True, temperature=True))
    simulation.step(100)
    del simulation

    sys_out = meta.System('output.pdb')

    assert sys_in.numAtoms == sys_out.numAtoms

    save_remove('output.pdb')

    return