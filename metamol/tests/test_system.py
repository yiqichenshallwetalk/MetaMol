import pytest
import os
from pkg_resources import resource_filename

import metamol as meta
from metamol.lib.molecules.alkane import *
from metamol.lib.molecules.polymer import *
from metamol.lib.fragments.monomers import *
from metamol.utils.help_functions import save_remove

def __get_param__(monomers_seq=[], head=[], tail=[], N=[]):
    for m_seq in monomers_seq:
        for h in head:
            for t in tail:
                for num in N:
                    yield (m_seq[0], m_seq[1], h, t, num)

@pytest.mark.parametrize('filename', 
['benzene.mol2', 'ethane.mol2', 'pf6.mol2',
'fused.mol2', 'peg100.mol2', 'fullerene.pdb', 
'styrene.mol2', 'cholesterol.pdb', 'Villin.pdb'])
def test_save_write(filename):
    name, ext_in = filename.split('.')
    file_in = resource_filename("metamol", os.path.join("tests", "files", filename))
    assert os.path.exists(file_in)
    sys = meta.System(file_in, name=name)

    extensions = ['.mol2', '.pdb', '.xyz']
    for ext in extensions:
        save_file = resource_filename("metamol", os.path.join("tests", "files", name+'_test'+ext))
        sys.save(save_file)
        sys_in = meta.System(save_file, name=name)

        assert sys.numAtoms == sys_in.numAtoms
        this_atoms = sorted([a.atomic for a in sys.atoms_iter()])
        other_atoms = sorted([a.atomic for a in sys_in.atoms_iter()])
        assert this_atoms == other_atoms
        if ext != '.xyz':
            assert np.allclose(sys.xyz, sys_in.xyz, atol=1.0e-3)
        save_remove(save_file)

@pytest.mark.parametrize('monomers,seq,head,tail,N', __get_param__(
    monomers_seq = [([CH2()], 'A'), ([PEGMonomer()], 'A'), ([CH2(), PEGMonomer()], 'AB')],
    head = [None, CH3(), OH()], tail = [None, CH3()],
    N = [1, 5, 10],
    )
)
def test_system_io(monomers, seq, head, tail, N):
    pol = Polymer(monomers, seq=seq, head=head, tail=tail)
    pol.build(N)
    pol.embed()

    ethane = Ethane()
    pol_sys = meta.System([pol, ethane], [60//N, 50], box=(60, 60, 60))
    pol_sys.parametrize(forcefield_name='opls')
    pol_sys.initial_config()

    extensions = ['.xyz', '.mol2', '.pdb', '.data', 'gromacs']
    for ext in extensions:
        if ext != 'gromacs':
            save_files = [resource_filename("metamol", os.path.join("tests", "files", 'temp'+ext))]
        else:
            save_files = [resource_filename("metamol", os.path.join("tests", "files", 'temp.gro')), 
                          resource_filename("metamol", os.path.join("tests", "files", 'temp.top'))]
        
        for save_file in save_files:
            pol_sys.save(save_file)
        pol_sys_in = meta.System()
        for save_file in save_files:
            pol_sys_in.readfile(filename=save_file)

        assert pol_sys.numAtoms == pol_sys_in.numAtoms
        assert np.allclose(pol_sys.xyz, pol_sys_in.xyz, atol=1.0e-3)

        if ext not in ('.xyz', '.pdb'):
            assert pol_sys.numMols == pol_sys_in.numMols
        if ext  in ('.data', 'gromacs'):
            assert pol_sys.numMols == pol_sys_in.numMols
            assert pol_sys.numBonds == pol_sys_in.numBonds
            assert pol_sys.numAngles == pol_sys_in.numAngles
            assert pol_sys.numDihedrals == pol_sys_in.numDihedrals
            assert pol_sys.numRBs == pol_sys_in.numRBs
            assert pol_sys.numImpropers == pol_sys_in.numImpropers
            assert len(pol_sys.params['atom_type']) == len(pol_sys_in.params['atom_type'])
            assert len(pol_sys.params['bond_type']) == len(pol_sys_in.params['bond_type'])
            assert len(pol_sys.params['angle_type']) == len(pol_sys_in.params['angle_type'])
            assert len(pol_sys.params['dihedral_type']) == len(pol_sys_in.params['dihedral_type'])
            assert len(pol_sys.params['rb_torsion_type']) == len(pol_sys_in.params['rb_torsion_type'])
            assert len(pol_sys.params['improper_type']) == len(pol_sys_in.params['improper_type'])
        for save_file in save_files:
            save_remove(save_file)

@pytest.mark.parametrize('monomers,seq,head,tail,N', __get_param__(
    monomers_seq = [([CH2(), PEGMonomer()], 'AB')],
    head = [None, CH3(), OH()], tail = [None, CH3()],
    N = [1, 5, 10],
    )
)
def test_system_water(monomers, seq, head, tail, N):
    pol = Polymer(monomers, seq=seq, head=head, tail=tail)
    pol.build(N)
    pol.embed()

    water = meta.SPCE()
    pol_sys = meta.System([pol, water], [60//N, 50], box=(60, 60, 60))
    pol_sys.parametrize(forcefield_name='opls')
    pol_sys.initial_config()

    extensions = ['.mol2', '.pdb', '.xyz', '.data', 'gromacs']
    for ext in extensions:
        if ext != 'gromacs':
            save_files = [resource_filename("metamol", os.path.join("tests", "files", 'temp'+ext))]
        else:
            save_files = [resource_filename("metamol", os.path.join("tests", "files", 'temp.gro')), 
                          resource_filename("metamol", os.path.join("tests", "files", 'temp.top'))]
        print(ext)        
        
        for save_file in save_files:
            pol_sys.save(save_file)
        pol_sys_in = meta.System()
        for save_file in save_files:
            pol_sys_in.readfile(filename=save_file)

        assert pol_sys.numAtoms == pol_sys_in.numAtoms
        assert np.allclose(pol_sys.xyz, pol_sys_in.xyz, atol=1.0e-3)

        if ext not in ('.xyz', '.pdb'):
            assert pol_sys.numMols == pol_sys_in.numMols
        if ext == '.data':
            assert pol_sys.numMols == pol_sys_in.numMols
            assert pol_sys.numBonds == pol_sys_in.numBonds
            assert pol_sys.numAngles == pol_sys_in.numAngles
            assert pol_sys.numDihedrals == pol_sys_in.numDihedrals
            assert pol_sys.numRBs == pol_sys_in.numRBs
            assert pol_sys.numImpropers == pol_sys_in.numImpropers
            assert len(pol_sys.params['atom_type']) == len(pol_sys_in.params['atom_type'])
            assert len(pol_sys.params['bond_type']) == len(pol_sys_in.params['bond_type'])
            assert len(pol_sys.params['angle_type']) == len(pol_sys_in.params['angle_type'])
            assert len(pol_sys.params['dihedral_type']) == len(pol_sys_in.params['dihedral_type'])
            assert len(pol_sys.params['rb_torsion_type']) == len(pol_sys_in.params['rb_torsion_type'])
            assert len(pol_sys.params['improper_type']) == len(pol_sys_in.params['improper_type'])
        for save_file in save_files:
            save_remove(save_file)