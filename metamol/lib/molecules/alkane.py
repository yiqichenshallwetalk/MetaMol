from typing import Iterable
import metamol as meta
from metamol.exceptions import MetaError
from metamol.lib.fragments import CH3, CH2

from copy import deepcopy

class Methane(meta.Molecule):
    """A Methane(CH4) molecule."""
    def __init__(self, align='x'):
        super(Methane, self).__init__(meta.Atom(atomic=6, resname='MET'), name='methane')

        H1 = meta.Atom(idx=2, atomic=1, resname='MET', x=1.087)
        H2 = meta.Atom(idx=3, atomic=1, resname='MET', x=-3.72060402e-01, z=-1.02134228)
        H3 = meta.Atom(idx=4, atomic=1, resname='MET', 
            x=-3.57469798e-01, y=8.90505097e-01, z=5.10671141e-01)
        H4 = meta.Atom(idx=5, atomic=1, resname='MET', 
            x=-3.57469798e-01, y=-8.90505097e-01, z=5.10671141e-01)
        
        self.atomList += [H1, H2, H3, H4]
        self.numAtoms += 4

        self.add_bond((0, 1))
        self.add_bond((0, 2))
        self.add_bond((0, 3))
        self.add_bond((0, 4))

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

class Ethane(meta.Molecule):
    """An Ethane(Ch3Ch3) Molecule."""  
    def __init__(self, align='x'):
        super(Ethane, self).__init__()

        ch3_head = CH3()
        ch3_tail = CH3()

        #Rotate the second CH3 for 180 degrees.
        ch3_tail.rotate(about='y', angle=180.0, rad=False)
        
        self.copy(ch3_head)
        self.connect(loc1=1, 
                    other=ch3_tail, 
                    loc2=1, 
                    keep_coords=True, 
                    orientation=[1, 0, 0])
        
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
        for atom in self.atoms_iter():
            atom.resname = 'ETH'
        self.name = 'ethane'

class Alkane(meta.Molecule):
    """An Alkane molecule with n carbons."""
    def __init__(self, N, align='x'):
        super(Alkane, self).__init__(name='alkane'+str(N))
        if N < 1:
            raise MetaError("Number of carbon in an alkane must be a positive integer")
        elif N == 1:
            self.copy(Methane(align=align))
            self.align = align
        elif N == 2:
            self.copy(Ethane(align=align))
            self.align = align
        else:
            ch3_head = CH3()
            frags = []
            for _ in range(N-2):
                frags.append(CH2())
            ch3_tail = CH3()
            ch3_tail.rotate(about='y', angle=180.0, rad=False)
            frags.append(ch3_tail)

            temp_mol = deepcopy(ch3_head)
            #self.copy(ch3_head)
            temp_mol.name = 'alkane' + str(N)
            loc, addon = 1, temp_mol.numAtoms
            for frag in frags:
                temp_mol.connect(loc1=loc, other=frag, 
                            loc2=1, keep_coords=True, 
                            orientation=[1, 0, 0])
                loc += addon
                addon = frag.numAtoms
            
            if align == 'y':
                temp_mol.rotate(about='z', angle=90.0, rad=False)

            elif align == 'z':
                temp_mol.rotate(about='y', angle=-90.0, rad=False)

            elif isinstance(align, Iterable):
                temp_mol.align_to(align)

            else:
                raise TypeError(
                    "align argument must be either a sting or an Iterable."
                    )

            self.copy(temp_mol)
            self.align = align

            for atom in self.atoms_iter():
                atom.resname = 'ALK'

            
