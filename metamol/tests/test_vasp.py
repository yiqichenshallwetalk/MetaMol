import pytest
import os
from pkg_resources import resource_filename
from distutils.spawn import find_executable
import numpy as np

import metamol as meta
from metamol.lib.molecules.alkane import *
from metamol.lib.molecules.benzene import *
from metamol.lib.molecules.water import *
from metamol.utils.help_functions import save_remove
from metamol.utils.execute import runCommands

class Test_Vasp:
    @pytest.fixture
    def gilmerite(self):
        #Gilmarite, Cu3(AsO4)(OH)3
        gilmerite_mol = meta.Molecule(name='gilmerite')
        gilmerite_mol.atomList.append(meta.Atom(idx=1, symbol="Cu", resname="Gil", x=0.0615697, y=2.8454988, z=1.4878373))
        gilmerite_mol.atomList.append(meta.Atom(idx=2, symbol="Cu", resname="Gil", x=2.8204174, y=1.3594167, z=1.6465892))
        gilmerite_mol.atomList.append(meta.Atom(idx=3, symbol="Cu", resname="Gil", x=2.8377297, y=4.3348074, z=1.7854972))

        gilmerite_mol.atomList.append(meta.Atom(idx=4, symbol="As", resname="Gil", x=5.43783569, y=1.54457900e-03, z=4.61488000e-04))
        gilmerite_mol.atomList.append(meta.Atom(idx=5, symbol="O", resname="Gil", x=0.8094841, y=1.4370937, z=0.3276565))
        gilmerite_mol.atomList.append(meta.Atom(idx=6, symbol="O", resname="Gil", x=0.8451646, y=4.5650253, z=0.3415011))
        gilmerite_mol.atomList.append(meta.Atom(idx=7, symbol="O", resname="Gil", x=3.6793642, y=2.8482277, z=0.7106915))
        gilmerite_mol.atomList.append(meta.Atom(idx=8, symbol="O", resname="Gil", x=3.9833167, y=-0.0501952, z=0.8583677))
        gilmerite_mol.atomList.append(meta.Atom(idx=9, symbol="O", resname="Gil", x=1.7415644, y=2.7864442, z=2.4828055))
        gilmerite_mol.atomList.append(meta.Atom(idx=10, symbol="O", resname="Gil", x=2.4149659, y=-0.0539473, z=2.9073744))
        gilmerite_mol.atomList.append(meta.Atom(idx=11, symbol="O", resname="Gil", x=4.5795937, y=3.6822546, z=3.0135167))

        gilmerite_mol.numAtoms = len(gilmerite_mol.atomList)
        
        gilmerite = meta.System(gilmerite_mol, box=[54.4500017, 58.7300015, 51.0400009], box_angle = [114.94999695, 93.05000305, 91.91999817])
        return gilmerite

    def test_vasp_rw_gilmerite(self, gilmerite):
        save_file = resource_filename("metamol", os.path.join("tests", "files", 'gil.poscar'))
        gilmerite.save(save_file)
        gilmerite_new = meta.System(save_file)

        assert np.allclose(gilmerite.box.lengths, gilmerite_new.box.lengths)
        assert np.allclose(gilmerite.box.angle, gilmerite_new.box.angle)
        assert np.allclose(gilmerite.numAtoms, gilmerite_new.numAtoms)
        
        atoms_ori = np.array([(atom.symbol, idx) for idx, atom in enumerate(gilmerite.atoms)], 
                     dtype=[('typ', '<U5'), ('idx', int)])
        xyz_ori = gilmerite.xyz[np.argsort(atoms_ori, order=('typ', 'idx'))]
        atoms_new = np.array([(atom.symbol, idx) for idx, atom in enumerate(gilmerite_new.atoms)], 
                     dtype=[('typ', '<U5'), ('idx', int)])
        xyz_new = gilmerite_new.xyz[np.argsort(atoms_new, order=('typ', 'idx'))]
        assert np.allclose(xyz_ori, xyz_new)

        save_remove(save_file)

    @pytest.mark.parametrize('mol', [Methane(), Ethane(), Benzene(), SPCE()])
    def test_vasp_rw_general(self, mol):
        save_file = resource_filename("metamol", os.path.join("tests", "files", 'test.poscar'))
        sys = meta.System(mol, dup=10, box=[30.0, 30.0, 30.0], 
                        box_angle=[114.94999695, 93.05000305, 91.91999817], name='test_system')
        sys.flatten()
        sys.save(save_file)
        sys_new = meta.System(save_file)

        assert np.allclose(sys.box.lengths, sys_new.box.lengths)
        assert np.allclose(sys.box.angle, sys_new.box.angle)
        assert np.allclose(sys.numAtoms, sys_new.numAtoms)
        
        atoms_ori = np.array([(atom.symbol, idx) for idx, atom in enumerate(sys.atoms)], 
                     dtype=[('typ', '<U5'), ('idx', int)])
        xyz_ori = sys.xyz[np.argsort(atoms_ori, order=('typ', 'idx'))]
        atoms_new = np.array([(atom.symbol, idx) for idx, atom in enumerate(sys_new.atoms)], 
                     dtype=[('typ', '<U5'), ('idx', int)])
        xyz_new = sys_new.xyz[np.argsort(atoms_new, order=('typ', 'idx'))]
        assert np.allclose(xyz_ori, xyz_new)

        sys = meta.System(mol, dup=10, box=[30.0, 30.0, 30.0], name='test_system')
        sys.flatten()
        sys.save(save_file)
        sys_new = meta.System(save_file)

        assert np.allclose(sys.box.lengths, sys_new.box.lengths)
        assert np.allclose(sys.box.angle, sys_new.box.angle)
        assert np.allclose(sys.numAtoms, sys_new.numAtoms)
        
        atoms_ori = np.array([(atom.symbol, idx) for idx, atom in enumerate(sys.atoms)], 
                     dtype=[('typ', '<U5'), ('idx', int)])
        xyz_ori = sys.xyz[np.argsort(atoms_ori, order=('typ', 'idx'))]
        atoms_new = np.array([(atom.symbol, idx) for idx, atom in enumerate(sys_new.atoms)], 
                     dtype=[('typ', '<U5'), ('idx', int)])
        xyz_new = sys_new.xyz[np.argsort(atoms_new, order=('typ', 'idx'))]
        assert np.allclose(xyz_ori, xyz_new)

        save_remove(save_file)

    VASP = find_executable("vasp_gpu") or find_executable("vasp_std")
    @pytest.mark.skipif(
        not VASP, reason="Vasp executable not installed or found"
    )
    @pytest.mark.parametrize('path', ['H', 'Si', 'Si128', 'siHugeShort', 'silicaIFPEN'])
    def test_vasp_execute(self, path):
        work_dir = resource_filename("metamol", os.path.join("tests", "files", "vasp", path))
        VASP = find_executable("vasp_gpu") or find_executable("vasp_std")
        rc, out, err = runCommands(cmds=VASP, raise_error=False, work_dir=work_dir)
        assert rc == 0