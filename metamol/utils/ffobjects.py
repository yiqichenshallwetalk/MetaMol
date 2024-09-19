
__all__ = ["Angle", "Dihedral", "RB_Torsion", "Improper", 
            "AtomType", "BondType", "AngleType", "DihedralType", 
            "RBTorsionType", "ImproperType"]

class Angle(object):
    def __init__(self, atom1=None, atom2=None, atom3=None, angleidx=-1):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.typeID = angleidx
    
    def __repr__(self):
        """Representation of the Angle."""

        desc = ["Angle id: {}".format(id(self))]
        if self.atom1:
            desc.append("Atom1: idx, {0}; type, {1}; symbol, {2}".format(self.atom1.idx, self.atom1.type, self.atom1.symbol))
            desc.append("Atom2: idx, {0}; type, {1}; symbol, {2}".format(self.atom2.idx, self.atom2.type, self.atom2.symbol))
            desc.append("Atom3: idx, {0}; type, {1}; symbol, {2}".format(self.atom3.idx, self.atom3.type, self.atom3.symbol))
            desc.append("Angle Type Index: {}".format(self.typeID))

        return "\n".join(desc)        

class Dihedral(object):
    def __init__(self, atom1=None, atom2=None, atom3=None, atom4=None, dtidx=-1):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.atom4 = atom4
        self.typeID = dtidx

    def __repr__(self):
        """Representation of the Dihedral."""

        desc = ["Dihedral id: {}".format(id(self))]
        if self.atom1:
            desc.append("Atom1: idx, {0}; type, {1}; symbol, {2}".format(self.atom1.idx, self.atom1.type, self.atom1.symbol))
            desc.append("Atom2: idx, {0}; type, {1}; symbol, {2}".format(self.atom2.idx, self.atom2.type, self.atom2.symbol))
            desc.append("Atom3: idx, {0}; type, {1}; symbol, {2}".format(self.atom3.idx, self.atom3.type, self.atom3.symbol))
            desc.append("Atom4: idx, {0}; type, {1}; symbol, {2}".format(self.atom4.idx, self.atom4.type, self.atom4.symbol))
            desc.append("Dihedral Type Index: {}".format(self.typeID))

        return "\n".join(desc)  

class RB_Torsion(object):
    def __init__(self, atom1=None, atom2=None, atom3=None, atom4=None, rbidx=-1):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.atom4 = atom4
        self.typeID = rbidx

    def __repr__(self):
        """Representation of the RB Style Torsion."""

        desc = ["RB Torsion id: {}".format(id(self))]
        if self.atom1:
            desc.append("Atom1: idx, {0}; type, {1}; symbol, {2}".format(self.atom1.idx, self.atom1.type, self.atom1.symbol))
            desc.append("Atom2: idx, {0}; type, {1}; symbol, {2}".format(self.atom2.idx, self.atom2.type, self.atom2.symbol))
            desc.append("Atom3: idx, {0}; type, {1}; symbol, {2}".format(self.atom3.idx, self.atom3.type, self.atom3.symbol))
            desc.append("Atom4: idx, {0}; type, {1}; symbol, {2}".format(self.atom4.idx, self.atom4.type, self.atom4.symbol))
            desc.append("RB Torsion Type Index: {}".format(self.typeID))

        return "\n".join(desc)  

class Improper(object):
    def __init__(self, atom1=None, atom2=None, atom3=None, atom4 = None, itidx=-1):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.atom4 = atom4
        self.typeID = itidx

    def __repr__(self):
        """Representation of the Improper Torsion."""

        desc = ["Improper id: {}".format(id(self))]
        if self.atom1:
            desc.append("Atom1: idx, {0}; type, {1}; symbol, {2}".format(self.atom1.idx, self.atom1.type, self.atom1.symbol))
            desc.append("Atom2: idx, {0}; type, {1}; symbol, {2}".format(self.atom2.idx, self.atom2.type, self.atom2.symbol))
            desc.append("Atom3: idx, {0}; type, {1}; symbol, {2}".format(self.atom3.idx, self.atom3.type, self.atom3.symbol))
            desc.append("Atom4: idx, {0}; type, {1}; symbol, {2}".format(self.atom4.idx, self.atom4.type, self.atom4.symbol))
            desc.append("Improper Type Index: {}".format(self.typeID))

        return "\n".join(desc)  

class AtomType(object):
    def __init__(self, idx=-1, name=None, atomic=0, symbol=None, sigma=0, epsilon=0, mass=0, charge=0):
        self.idx = idx
        self.name = name
        self.atomic = atomic
        self.symbol = symbol
        self.sigma = sigma
        self.epsilon = epsilon
        self.mass = mass
        self.charge = charge
    
    def __repr__(self):
        desc = ["Atom Type id: {}".format(id(self))]
        if self.idx > 0:
            if self.name:
                desc.append("Type Name: {0}".format(self.name))
            desc.append("Type Index: {0}".format(self.idx))
            desc.append("Symbol: {0}".format(self.symbol))
            desc.append("Mass: {0}".format(self.mass))
            desc.append("Sigma: {0} Angstrom".format(self.sigma))
            desc.append("Epsilon: {0} kcal/mol".format(self.epsilon))
            desc.append("charge: {0} e".format(self.charge))

        return "\n".join(desc)  

class BondType(object):
    def __init__(self, idx=-1, name=None, k=0, req=0):
        self.idx = idx
        self.name = name
        self.k = k
        self.req = req
    
    def __repr__(self):
        desc = ["Bond Type id: {}".format(id(self))]
        if self.idx > 0:
            desc.append("Type Name: {0}".format(self.name))
            desc.append("Type Index: {0}".format(self.idx))
            desc.append("k: {0} kcal/mol/Angstrom^2".format(self.k))
            desc.append("req: {0} Angstrom".format(self.req))

        return "\n".join(desc) 

class AngleType(object):
    def __init__(self, idx=-1, name=None, k=0, theteq=0, ubk=0, ubreq=0):
        self.idx = idx
        self.name = name
        self.k = k
        self.theteq = theteq
        self.ubk = ubk
        self.ubreq = ubreq
    
    def __repr__(self):
        desc = ["Angle Type id: {}".format(id(self))]
        if self.idx > 0:
            desc.append("Type Name: {0}".format(self.name))
            desc.append("Type Index: {0}".format(self.idx))
            desc.append("k: {0} kcal/mol/rad^2".format(self.k))
            desc.append("theteq: {0} degrees".format(self.theteq))

        return "\n".join(desc) 

class DihedralType(object):
    def __init__(self, idx=-1, name=None, phi_k=0, per=0, phase=0):
        self.idx = idx
        self.name = name
        self.phi_k = phi_k
        self.per = per
        self.phase = phase
    
    def __repr__(self):
        desc = ["Dihedral Type id: {}".format(id(self))]
        if self.idx > 0:
            desc.append("Type Name: {0}".format(self.name))
            desc.append("Type Index: {0}".format(self.idx))
            desc.append("phi_k: {0} kcal/mol".format(self.phi_k))
            desc.append("per: {0}".format(self.per))
            desc.append("phase: {0} degrees".format(self.phase))

        return "\n".join(desc) 

class RBTorsionType(object):
    def __init__(self, idx=-1, name=None, c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, opls=False):
        self.idx = idx
        self.name = name
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
        self.opls = opls
    
    def __repr__(self):
        desc = ["RB Torsion Type id: {}".format(id(self))]
        if self.idx > 0:
            desc.append("Type Name: {0}".format(self.name))
            desc.append("Type Index: {0}".format(self.idx))
            if self.opls:
                desc.append("Style: OPLS")
            else:
                desc.append("Style: RB")
                desc.append("c0: {0} kcal/mol".format(self.c0))
            desc.append("c1: {0} kcal/mol".format(self.c1))
            desc.append("c2: {0} kcal/mol".format(self.c2))
            desc.append("c3: {0} kcal/mol".format(self.c3))
            desc.append("c4: {0} kcal/mol".format(self.c4))
            desc.append("c5: {0} kcal/mol".format(self.c5))

        return "\n".join(desc) 

class ImproperType(object):
    def __init__(self, idx=None, name=None, psi_k=None, psi_eq=None):
        self.idx = idx
        self.name = name
        self.phi_k = psi_k
        self.per = psi_eq
    
    def __repr__(self):
        desc = ["Improper Type id: {}".format(id(self))]
        if self.idx > 0:
            desc.append("Type Name: {0}".format(self.name))
            desc.append("Type Index: {0}".format(self.idx))
            desc.append("psi_k: {0} kcal/mol/rad^2".format(self.phi_k))
            desc.append("psi_eq: {0} degrees".format(self.per))

        return "\n".join(desc) 
