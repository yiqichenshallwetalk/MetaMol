from typing import Iterable

import metamol as meta

class Benzene(meta.Molecule):
    """An benzene molecule."""
    def __init__(self, align='x'):
        super(Benzene, self).__init__(name='Benzene')
        C1 = meta.Atom(idx=1, atomic=6, resname='BEN', x=-1.394)
        C2 = meta.Atom(idx=2, atomic=6, resname='BEN', x=-0.697, y=1.207)
        C3 = meta.Atom(idx=3, atomic=6, resname='BEN', x=0.697, y=1.207)
        C4 = meta.Atom(idx=4, atomic=6, resname='BEN', x=1.394)
        C5 = meta.Atom(idx=5, atomic=6, resname='BEN', x=0.697, y=-1.207)
        C6 = meta.Atom(idx=6, atomic=6, resname='BEN', x=-0.697, y=-1.207)
        H1 = meta.Atom(idx=7, atomic=1, resname='BEN', x=-2.478)
        H2 = meta.Atom(idx=8, atomic=1, resname='BEN', x=-1.239, y=2.146)
        H3 = meta.Atom(idx=9, atomic=1, resname='BEN', x=1.239, y=2.146)
        H4 = meta.Atom(idx=10, atomic=1, resname='BEN', x=2.478)
        H5 = meta.Atom(idx=11, atomic=1, resname='BEN', x=1.239, y=-2.146)
        H6 = meta.Atom(idx=12, atomic=1, resname='BEN', x=-1.239, y=-2.146)

        self.atomList += [C1, C2, C3, C4, C5, C6, H1, H2, H3, H4, H5, H6]
        self.numAtoms += 12

        self.add_bond((0, 1))
        self.add_bond((0, 6))
        self.add_bond((1, 7))
        self.add_bond((1, 2))
        self.add_bond((2, 8))
        self.add_bond((2, 3))
        self.add_bond((3, 9))
        self.add_bond((3, 4))
        self.add_bond((4, 10))
        self.add_bond((4, 5))
        self.add_bond((5, 11))
        self.add_bond((5, 0))

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