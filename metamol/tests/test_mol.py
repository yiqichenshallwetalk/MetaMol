import pytest
import numpy as np
import os
from pkg_resources import resource_filename

from rdkit import Chem

import metamol as meta
from metamol.utils.help_functions import distance, save_remove

@pytest.mark.parametrize('smi', 
['c1cc(CCCO)ccc1', 'c1cc(CCCO)ccc1', 
'Cc1c(COc2cc(OCc3cccc(c3)C#N)c(CN3C[C@H](O)C[C@H]3C(O)=O)cc2Cl)cccc1-c1ccc2OCCOc2c1',
'CC(NCCNCC1=CC=C(OCC2=C(C)C(C3=CC=CC=C3)=CC=C2)N=C1OC)=O'])
def test_smi_input(smi):
    mmol = meta.Molecule(smi, smiles=True)
    rdmol = Chem.MolFromSmiles(smi)
    rdmol = Chem.AddHs(rdmol)
    assert mmol.numAtoms == rdmol.GetNumAtoms()
    assert mmol.numBonds == rdmol.GetNumBonds()

    bonds_mm, bonds_rd = [], []
    for bond in mmol.bonds_iter():
        bonds_mm.append(sorted((bond[0].idx, bond[1].idx)))
    for bond in rdmol.GetBonds():
        bonds_rd.append(sorted((bond.GetBeginAtomIdx()+1, bond.GetEndAtomIdx()+1)))

    assert sorted(bonds_mm) == sorted(bonds_rd)

    return
            
@pytest.mark.parametrize('smi_string', 
['c1cc(CCCO)ccc1'])
def test_add_remove(smi_string):
    mol_in = meta.Molecule(smi_string, smiles=True)
    remove_O = [a for a in mol_in.atoms_iter() if a.atomic==8]

    connect_ports = []
    for a in remove_O:
        for neigh in mol_in.neighbors[a]:
            if neigh.atomic != 1:
                connect_ports.append(neigh)
        mol_in.remove_atom(a)

    return

@pytest.mark.parametrize('about', [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)])
@pytest.mark.parametrize('angle', [np.pi/2.0, np.pi/4.0, np.pi/8.0])
def test_rotate(about, angle):
    from metamol.lib.fragments.monomers import PEGMonomer
    from metamol.lib.molecules.polymer import Polymer

    PEG = Polymer(monomers=[PEGMonomer()], name='PEG')
    PEG.build(N=5)

    xyz_original = PEG.xyz
    PEG.rotate(about, angle)
    PEG.rotate(about, -angle)
    xyz_new = PEG.xyz
    assert np.allclose(xyz_original, xyz_new)

    PEG.rotate(about, angle)
    PEG.rotate(tuple([-a for a in about]), angle)
    xyz_new = PEG.xyz
    assert np.allclose(xyz_original, xyz_new)

    return
