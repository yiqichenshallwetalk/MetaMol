from typing import Iterable
import numpy as np
from ele import (element_from_atomic_number,
    element_from_symbol,
    element_from_name,
    element_from_mass)

from metamol.exceptions import MetaError

class Atom(object):

    def __init__(self, idx=1, chain=1, atomic=-1, symbol="", name="", mass=0.0, resname="", resid=-1, x=0, y=0, z=0, charge=0.0, atomtype=None, atidx=-1):
        super(Atom, self).__init__()
        
        if not isinstance(atomic, int): 
            raise ValueError("Atomic number must be an integer")
        elif atomic==0:
            element = None
        elif atomic>0:
            try:
                element = element_from_atomic_number(atomic)
            except:
                element = None
        elif symbol:
            try:
                element = element_from_symbol(symbol)
            except:
                element = None
        # elif name:
        #     try:
        #         element = element_from_name(name)
        #     except:
        #         element = None
        elif mass>0:
            try:
                element = element_from_mass(mass)
            except:
                element = None
        else:
            element = None

        self.idx = idx
        self.chain = chain
        self.x = x
        self.y = y
        self.z = z
        self.resname=resname
        self.resid = resid
        self.charge = charge
        self._mass = 0.0
        self.name = name

        if element is None:
            if atomic == 0:
                self.atomic = 0
                self.symbol = 'D'
            else:
                self.atomic = atomic
                self.symbol = symbol
            # if not name:
            #     self.name = 'UNL'
            # else:
            #     self.name = name
            
        else:
            self.atomic = element.atomic_number
            self.symbol = element.symbol
            # if not name:
            #     self.name = element.symbol
            # else:
            #     self.name = name
            if not mass:
                self._mass = element.mass
            else:
                self._mass = mass

        # FF AtomType
        self.type = atomtype
        self.atidx = atidx

        self._check_input_sanity()

    def _check_input_sanity(self):
        """Check the sanity of input values."""
        # TODO
        pass

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, m):
        if not isinstance(m, (int, float)):
            raise TypeError("mass must be a numeric value.")
        if m < 0.0:
            raise TypeError("mass cannot be negative.")
        self._mass = m

    @property
    def xyz(self):
        return np.asarray([self.x, self.y, self.z])

    @xyz.setter
    def xyz(self, coords):
        if not isinstance(coords, Iterable):
            raise TypeError("The new coordinates must be an Iterable")
        if len(coords) != 3:
            raise MetaError("The length of the new coordinates for an Atom must be 3")
        
        self.x = coords[0]
        self.y = coords[1]
        self.z = coords[2]

    def update_idx(self, idx):
        self.idx = idx
    
    def __repr__(self):
        """Representation of the Atom."""

        desc = ["Atom id: {}".format(id(self))]
        desc.append("Name: {}".format(self.name))
        desc.append("Symbol: {}".format(self.symbol))
        desc.append("Atomic Number: {}".format(self.atomic))
        desc.append("Atomic Mass: {}".format(self.mass))
        desc.append("Atomic Charge: {}".format(self.charge))
        desc.append("Atom index in the System: {}".format(self.idx))
        if self.type:
            desc.append("Atom Type: {}".format(self.type))
        desc.append("Coordinates: [{0}, {1}, {2}]".format(self.x, self.y, self.z))

        return "\n".join(desc)