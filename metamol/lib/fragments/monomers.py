"""Lib of frequently used monomers"""
from typing import Iterable
import numpy as np
from bisect import bisect

import metamol as meta
#from metamol.atom import Atom
#from metamol.utils.geometry import RotateX, RotateY, RotateZ

class CH3(meta.Molecule):
    """A Methyl group."""
    def __init__(self, align='x'):
        super(CH3, self).__init__(meta.Atom(atomic=6, resname='Methyl'), name='Methyl')

        H1 = meta.Atom(idx=2, atomic=1, resname='Methyl', x=-3.72060402e-01, z=-1.02134228)
        H2 = meta.Atom(idx=3, atomic=1, resname='Methyl', 
            x=-3.57469798e-01, y=8.90505097e-01, z=5.10671141e-01)
        H3 = meta.Atom(idx=4, atomic=1, resname='Methyl', 
            x=-3.57469798e-01, y=-8.90505097e-01, z=5.10671141e-01)
        
        self.atomList += [H1, H2, H3]
        self.numAtoms += 3

        self.add_bond((0, 1))
        self.add_bond((0, 2))
        self.add_bond((0, 3))

        if align == 'y':
            self.rotate(about='z', angle=90.0, rad=False)

        elif align == 'z':
            self.rotate(about='y', angle=-90.0, rad=False)
        
        elif isinstance(align, Iterable):
            self.align_to(align)

        else:
            raise TypeError(
                "align argument must be either a sting or an Iterable."
                )

        self.align = align

        #End groups only have tail attribute, head is None
        self.head = None
        self.tail = 1

class CH2(meta.Molecule):
    """A Methylene group."""
    def __init__(self, align='x'):
        super(CH2, self).__init__(meta.Atom(atomic=6, resname='Methylene'), name='Methylene')

        ori_dict = {'x': ([0, 0, 1], [0, 0, -1]),
                    'y': ([0, 0, 1], [0, 0, -1]),
                    'z': ([1, 0, 0], [-1, 0, 0])
                }

        self.align = align
        self.add_atom(meta.Atom(atomic=1, resname='Methylene'), connect_to=1, bondlength=1.087, orientation=ori_dict[align][0])
        self.add_atom(meta.Atom(atomic=1, resname='Methylene'), connect_to=1, bondlength=1.087, orientation=ori_dict[align][1])

        self.head = 1
        self.tail = 1

class OH(meta.Molecule):
    """A hydroxyl group."""
    def __init__(self, align='x'):
        super(OH, self).__init__(meta.Atom(atomic=8, resname='hydroxyl'), name='hydroxyl')

        H = meta.Atom(idx=2, atomic=1, resname='hydroxyl', x=-1.10)
        self.atomList.append(H)
        self.numAtoms += 1

        self.add_bond((0, 1))   

        if align == 'y':
            self.rotate(about='z', angle=90.0, rad=False)

        elif align == 'z':
            self.rotate(about='y', angle=-90.0, rad=False)

        elif isinstance(align, Iterable):
            self.align_to(align)

        else:
            raise TypeError(
                "align argument must be either a sting or an Iterable."
                )
                
        self.align = align

        self.head = None
        self.tail = 1

class PEGMonomer(meta.Molecule):
    """A polyethylene glycol (PEG) monomer."""
    def __init__(self, align='x'):
        super(PEGMonomer, self).__init__(meta.Atom(atomic=6, resname='PEG'), name='PEG(Polyethylene Glycol)')

        H1 = meta.Atom(idx=2, atomic=1, resname='PEG', y=1.20)
        H2 = meta.Atom(idx=3, atomic=1, resname='PEG', y=-1.20)
        C2 = meta.Atom(idx=4, atomic=6, resname='PEG', x=1.54)
        H3 = meta.Atom(idx=5, atomic=1, resname='PEG', x=1.54, y=1.20)
        H4 = meta.Atom(idx=6, atomic=1, resname='PEG', x=1.54, y=-1.20)
        O1 = meta.Atom(idx=7, atomic=8, resname='PEG', x=2.97)

        self.atomList += [H1, H2, C2, H3, H4, O1]
        self.numAtoms += 6

        self.add_bond((0, 1))
        self.add_bond((0, 2))
        self.add_bond((0, 3))
        self.add_bond((3, 4))
        self.add_bond((3, 5))
        self.add_bond((3, 6))

        if align == 'y':
            self.rotate(about='z', angle=90.0, rad=False)

        elif align == 'z':
            self.rotate(about='y', angle=-90.0, rad=False)

        elif isinstance(align, Iterable):
            self.align_to(align)

        else:
            raise TypeError(
                "align argument must be either a sting or an Iterable."
                )
                
        self.align = align

        self.head = 1
        self.tail = 7

class Monomer(meta.Molecule):
    """A general class for monomers"""
    def __init__(self, mol, head=None, tail=None, align='x', **kwargs):
        if isinstance(mol, str):
            super(Monomer, self).__init__(input=mol, smiles=True)

        elif isinstance(mol, meta.Molecule):
            super(Monomer, self).__init__()
            self.copy(target=mol)
        else:
            raise TypeError("Can only create Monomer from smiles string or a meta.Molecule instance.")

        remove_atoms = kwargs.get('remove_atoms', [])
        remove_atoms = sorted(remove_atoms)
        for ra_idx in remove_atoms[::-1]:
            self.remove_atom(ra_idx)

        self.head = head
        self.tail = tail
        #self.embed()

        if head is None:
            avg_coords = np.average(self.xyz, axis=0)
            head_atom = meta.Atom(atomic=1, x=avg_coords[0], y= avg_coords[1], z=avg_coords[2])
        else:
            head -= bisect(remove_atoms, head)
            head_atom = self[head-1]

        if tail is None:
            avg_coords = np.average(self.xyz, axis=0)
            tail_atom = meta.Atom(atomic=1, x=avg_coords[0], y=avg_coords[1], z=avg_coords[2])
        else:
            tail -= bisect(remove_atoms, tail)
            tail_atom = self[tail-1]
        align_ori = [tail_atom.x-head_atom.x, tail_atom.y-head_atom.y, tail_atom.z-head_atom.z]
        self.align = align_ori
        self.align_to(align)

