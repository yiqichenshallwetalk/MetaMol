import numpy as np

import metamol as meta


class Water3Site(meta.Molecule):
    """A generic 3 site water model."""
    def __init__(self, oh_bond_length, angle, molname='Water-3site', resname='W3S'):
        super().__init__(name=molname)
        o1 = meta.Atom(idx=1, atomic=8, name='OW', resname=resname)
        h1 = meta.Atom(idx=2, atomic=1, name='HW1', resname=resname, x=oh_bond_length)
        h2 = meta.Atom(
            idx=3,
            atomic=1, 
            name='HW2',
            resname=resname, 
            x=oh_bond_length * np.cos(np.radians(angle)),
            y=oh_bond_length * np.sin(np.radians(180.0 - angle)),
            )
        
        #self.name = resname
        self.smi = 'O'
        self.atomList = [o1, h1, h2]
        self.numAtoms = 3
        self.add_bond((o1, h1))
        self.add_bond((o1, h2))

class TIP3P(Water3Site):
    """TIP3P water.
    Reference:  https://doi.org/10.1063/1.445869
    """

    def __init__(self):
        oh_bond_length = 0.9572
        angle = 104.52
        super().__init__(oh_bond_length, angle, molname='Water-TIP3P', resname='TIP3')

class SPCE(Water3Site):
    """SPCE water.
    Reference: https://doi.org/10.1021/j100308a038
    """

    def __init__(self):
        bond_length = 1.0
        angle = 109.47
        super().__init__(bond_length, angle, molname='Water-SPCE', resname='SPCE')

class Water4Site(meta.Molecule):
    """A generic 4-site water model."""
    def __init__(self, oh_bond_length, angle, om_bond_length, molname='Water-4site', resname='W4S'):
        super().__init__(name=molname)

        o1 = meta.Atom(idx=1, atomic=8, name='OW', resname=resname)
        h1 = meta.Atom(idx=2, atomic=1, name='HW1', resname=resname, x=oh_bond_length)
        h2 = meta.Atom(
            idx=3,
            atomic=1, 
            name='HW2',
            resname=resname, 
            x=oh_bond_length * np.cos(np.radians(angle)),
            y=oh_bond_length * np.sin(np.radians(180.0 - angle)),
            )
        m1 = meta.Atom(
            atomic=0,
            name='MW',
            x=om_bond_length * np.cos(np.radians(angle / 2.0)),
            y=om_bond_length * np.sin(np.radians(angle / 2.0)),
        )

        #self.name = resname
        self.smi = 'O'
        self.atomList = [o1, h1, h2, m1]
        self.numAtoms = 4
        self.add_bond((o1, h1))
        self.add_bond((o1, h2))

class TIP4P(Water4Site):
    """TIP4P water.
    Reference:  https://doi.org/10.1063/1.445869
    """

    def __init__(self):
        oh_bond_length = 0.9572
        om_bond_length = 0.15
        angle = 104.52
        super().__init__(oh_bond_length, angle, om_bond_length, molname='Water-TIP4P', resname='TIP4')

class TIP4P2005(Water4Site):
    """TIP4P/2005 water.
    Reference:  https://doi.org/10.1063/1.2121687
    """

    def __init__(self):
        oh_bond_length = 0.9572
        om_bond_length = 0.1546
        angle = 104.52
        super().__init__(oh_bond_length, angle, om_bond_length, molname='Water-TIP4P/2005', resname='TP05')