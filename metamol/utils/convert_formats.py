import warnings
import numpy as np

import metamol as meta
from metamol.system import Box
from metamol.exceptions import MetaError

__all__ = ["convert_from_openmm", "convert_from_rd", "mol_convert_from_rd", "convert_from_pmd",
            "convert_to_rd", "convert_mol_to_rd", "convert_to_pmd", "convert_to_openmm", "gromacs_to_openmm",
            "RB_to_OPLS", "OPLS_to_RB", "vectors_to_box", "box_to_vectors", "normalize_vectors",
            "reduced_from_vectors"]

def convert_from_openmm(topology, positions=None, asSystem=False, host_obj=None):
    """Convert an OpenMM Topology and Positions to a System object"""
    from openmm import unit as u
    from collections import defaultdict

    chain_bond_dict = defaultdict(list)

    for bond in topology.bonds():
        cid = bond[0].residue.chain.index
        chain_bond_dict[cid].append(bond)

    sys_out = meta.System()

    for chain in topology.chains():
        mol_to_add = meta.Molecule()
        start_atom_index = next(chain.atoms()).index
        for atom in chain.atoms():
            atom_to_add = meta.Atom(atom.index+1, atomic=atom.element.atomic_number, 
                resname=atom.residue.name+'.'+str(atom.residue.index+1), x=0.0, y=0.0, z=0.0)
            mol_to_add.atomList.append(atom_to_add)
            mol_to_add.numAtoms += 1

        # Add Bonds to the Molecule
        for bond in chain_bond_dict[chain.index]:
            mol_to_add.add_bond((bond[0].index-start_atom_index, 
                                bond[1].index-start_atom_index))

        sys_out.add(mol_to_add)

    if positions:
        positions = positions.value_in_unit(u.Unit({u.angstrom_base_unit: 1.0}))
        positions = [[p.x, p.y, p.z] for p in positions]
        sys_out.xyz = positions

    if not asSystem:
        if host_obj is not None:
            host_obj.copy(sys_out.molecules[0])
            return host_obj
        else:
            return sys_out.molecules[0]

    else:
        vecs = topology.getPeriodicBoxVectors()
        if vecs is not None:
            #Convert vectors units to angstroms
            vecs = vecs.value_in_unit(u.Unit({u.angstrom_base_unit: 1.0}))
            vecs = [[v.x, v.y, v.z] for v in vecs]
            box_lengths, box_angle = vectors_to_box(vecs)
            box_bounds = np.asarray([0.0, 0.0, 0.0] + list(box_lengths))
            sys_out.box = Box(bounds=box_bounds, angle=box_angle)
            
        if host_obj is not None:
            host_obj.copy(sys_out)
            return host_obj
        else:
            return sys_out

def convert_from_rd(rdmol, host_obj=None, asSystem=False, smi=None):
    """Convert a rdkit Mol object to a Molecule/System object."""
    from rdkit import Chem

    rdmol_tuple = Chem.GetMolFrags(rdmol, asMols=True)
    if not asSystem:
        return mol_convert_from_rd(rdmol_tuple[0], host_obj=host_obj, smi=smi)

    if not host_obj:
        sys_out = meta.System()
    else:
        sys_out = host_obj
    
    for rdm in rdmol_tuple:
        sys_out.add(mol_convert_from_rd(rdm, smi=smi))
    sys_out.update_atom_idx()
    return sys_out

def mol_convert_from_rd(rdmol, host_obj=None, smi=None):
    """Convert a rdkit Mol to a Molecule object."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol_out = meta.Molecule()
    
    if smi:
        mol_out.smi = smi
    else:
        mol_out.smi = Chem.MolToSmiles(Chem.RemoveHs(rdmol))

    if len(rdmol.GetConformers())==0:
        rdmol = Chem.AddHs(rdmol)
        if AllChem.EmbedMolecule(rdmol) != 0:
            raise MetaError("Rdkit is unable to generate 3D coordinates for {}".format(rdmol))
        
        AllChem.UFFOptimizeMolecule(rdmol)

    pos = rdmol.GetConformers()[0].GetPositions()
    for idx, a in enumerate(rdmol.GetAtoms()):
        a_temp = meta.Atom(idx=idx+1, atomic=a.GetAtomicNum(), 
            x=pos[idx][0], y=pos[idx][1], z=pos[idx][2], resname=mol_out.name)
        mol_out.atomList.append(a_temp)
    mol_out.numAtoms = idx + 1

    for idx, bond in enumerate(rdmol.GetBonds()):
        mol_out.add_bond((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    mol_out.numBonds = idx + 1

    if host_obj:
        host_obj.copy(mol_out)
    return mol_out

def convert_from_pmd(struct, host_obj=None, asSystem=True, xyz=None):
    """Convert a ParmEd Structure to a Molecule/System object"""
    from collections import defaultdict

    if len(struct.bonds) > 0 and struct.bonds[0].type is not None:
        parametrized = True
    else:
        parametrized = False

    topology = struct.topology
    chain_bond_dict = defaultdict(list)

    for bond in topology.bonds():
        cid = bond[0].residue.chain.index
        chain_bond_dict[cid].append(bond)

    sys_out = meta.System()

    for chain in topology.chains():
        set_chain, set_bonded = set(), set()
        chain_to_add = meta.Molecule(chain=int(chain.id))
        start_atom_index = next(chain.atoms()).index
        for atom in chain.atoms():
            set_chain.add(atom.index-start_atom_index)
            atomic_num = 0 if atom.element is None else atom.element.atomic_number
            atom_to_add = meta.Atom(atom.index+1, chain=int(chain.id), atomic=atomic_num, name=atom.name,
                resname=atom.residue.name, resid=atom.residue.index+1, x=0.0, y=0.0, z=0.0)
            chain_to_add.atomList.append(atom_to_add)
            chain_to_add.numAtoms += 1

        # Add Bonds to the Molecule
        for bond in chain_bond_dict[chain.index]:
            set_bonded.add((bond[0].index-start_atom_index))
            set_bonded.add((bond[1].index-start_atom_index))
            chain_to_add.add_bond((bond[0].index-start_atom_index, 
                                bond[1].index-start_atom_index))
        
        if chain_to_add.numBonds==0:
            sys_out.add(chain_to_add, update_index=False, assign_residues=False)
        else:
            # Break the chain into molecules
            mols = chain_to_add.bond_graph.connected_nodes()
            for mol in mols:
                mol = sorted(mol, key=lambda x: x.idx)
                mol_to_add = meta.Molecule(chain=int(chain.id))
                mol_to_add.atomList = mol
                mol_to_add.numAtoms = len(mol)
                for atom1 in mol:
                    for atom2 in chain_to_add.bond_graph.neighbors(atom1):
                        mol_to_add.add_bond((atom1, atom2))

                sys_out.add(mol_to_add, update_index=False, assign_residues=False)

            # Add nonbonded atoms as separate molecules
            nonbonded_atoms = list(set_chain-set_bonded)
            for atomidx in nonbonded_atoms:
                mol_to_add = meta.Molecule(input=chain_to_add[atomidx], chain=int(chain.id))
                sys_out.add(mol_to_add, update_index=False, assign_residues=False)

    del chain_to_add

    sys_out.numBonds = len(sys_out.bonds)
    sys_out.molList = sorted(sys_out.molList, key=lambda mol: mol[0].idx)
    sys_out.count_residues()
    assert sys_out.numAtoms == topology.getNumAtoms()
    assert sys_out.numBonds == topology.getNumBonds()

    if xyz is not None:
        sys_out.xyz = xyz
    elif struct.coordinates is not None:
        sys_out.xyz = struct.coordinates

    # atom_map = dict()

    # chain_atom_dict = defaultdict(list)
    # chain_bond_dict = defaultdict(list)

    # for residue in struct.residues:
    #     chain_atom_dict[residue.idx] += residue.atoms

    # for bond in struct.bonds:
    #     for idx, atoms in chain_atom_dict.items():
    #         if bond.atom1 in atoms:
    #             chain_bond_dict[idx].append(bond)
    #             break

    # for chain, atoms in chain_atom_dict.items():
    #     mol_to_add = meta.Molecule()
    #     for idx, atom in enumerate(atoms):
    #         atom_to_add = meta.Atom(idx+1, atomic=atom.atomic_number, 
    #             resname=atom.residue.name, x=atom.xx, y=atom.xy, z=atom.xz)
    #         atom_map[atom] = atom_to_add
    #         mol_to_add.atomList.append(atom_to_add)
    #         mol_to_add.numAtoms += 1

    #     # Add Bonds to the Molecule
    #     for bond in chain_bond_dict[chain]:
    #         mol_to_add.add_bond((atom_map[bond.atom1], atom_map[bond.atom2]))

    #     sys_out.add(mol_to_add)

    if parametrized:
        sys_out.parametrize(struct=struct)

    if not asSystem:
        if host_obj is not None:
            host_obj.copy(sys_out.molecules[0])
            return host_obj
        else:
            return sys_out.molecules[0]
    else:
        if struct.box is not None:
            box_lengths, box_angle = struct.box[:3], struct.box[3:]
            box_bounds = np.asarray([0.0, 0.0, 0.0] + list(box_lengths))
            sys_out.box = Box(bounds=box_bounds, angle=box_angle)
            
        if host_obj is not None:
            host_obj.copy(sys_out)
            return host_obj
        else:
            return sys_out

def convert_to_rd(obj):
    """Convert a Molecule/System object to a rdkit Mol object."""
    from rdkit import Chem

    if isinstance(obj, meta.Molecule):
        return convert_mol_to_rd(obj)
    elif isinstance(obj, meta.System):
        rdmol = Chem.Mol()
        for mol in obj.molecules_iter():
            rdmol_temp = convert_mol_to_rd(mol)
            rdmol = Chem.CombineMols(rdmol, rdmol_temp)
        return rdmol
    else:
        raise TypeError("The input object must be either a Molecule or a System")

def convert_mol_to_rd(mol):
    """Convert a Molecule object to a rdkit Mol object."""

    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    from metamol.utils.help_functions import fix_bond_order

    if not isinstance(mol, meta.Molecule):
        raise TypeError("The input object must be a Molecule")

    em = Chem.RWMol()
    startidx = mol[0].idx
    for i in range(mol.numAtoms):
        em.AddAtom(Chem.Atom(mol.atoms[i].atomic))
    for bond in mol.bonds_iter():
        em.AddBond(bond[0].idx-startidx, bond[1].idx-startidx, Chem.BondType.SINGLE)
    m = em.GetMol()

    #Correct bond order
    fix_bond_order(m)

    Chem.SanitizeMol(m)
    AllChem.EmbedMolecule(m, useRandomCoords=True)
    #AllChem.MMFFOptimizeMolecule(m)

    conf = m.GetConformer()
    for i in range(mol.numAtoms):
        xyz = mol.atoms[i].xyz
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    return m

def convert_to_pmd(obj, box=None, title="", residues=None, parametrize=False, bonds=True):
    """Convert a Molecule/System object to a ParmEd Structure"""
    import parmed as pmd
    struct = pmd.Structure()
    struct.title = title if title else obj.name

    if isinstance(obj, meta.Molecule):
        mols = [obj]
    elif isinstance(obj, meta.System):
        obj.flatten()
        mols = obj.molecules
    else:
        raise TypeError("The input object must be either a Molecule or a System")

    atom_map = dict()

    if not residues:
        residues = set()
        for atom in obj.atoms_iter():
            residues.add(atom.resname+str(atom.resid))

    if isinstance(residues, str):
        residues = [residues]
    if isinstance(residues, (list, set)):
        residues = tuple(residues)
    
    atom_res_map = dict()

    for idx, mol in enumerate(mols):
        atom_types = dict()
        for atom in mol.atoms_iter():
            chainid = idx+1 if atom.chain is None else atom.chain
            if atom.resname+str(atom.resid) not in atom_res_map:
                if residues and atom.resname+str(atom.resid) in residues:
                    current_residue = pmd.Residue(atom.resname, number=atom.resid, chain=str(chainid))
                else:
                    current_residue = pmd.Residue('RES', number=-1, chain=str(chainid))
                atom_res_map[atom.resname+str(atom.resid)] = current_residue

            atidx = atom_types.get(atom.symbol, 1)
            if atom.atomic==0:
                pmd_atom = pmd.Atom(atomic_number=0, name="UNL"+str(atidx), mass=0, charge=0)
            else:
                atomname = atom.name if atom.name else atom.symbol+str(atidx)
                #atomname = atom.name if atom.name else atom.symbol
                pmd_atom = pmd.Atom(
                    atomic_number=atom.atomic,
                    name=atomname,
                    type=atom.symbol,
                    mass=atom.mass,
                    charge=atom.charge
                )
                
            pmd_atom.xx, pmd_atom.xy, pmd_atom.xz = atom.x, atom.y, atom.z
            struct.add_atom(pmd_atom, resname=current_residue.name, resnum=current_residue.number, chain=str(chainid))
            #atom_map[atom] = pmd_atom
            atom_map[atom.idx] = pmd_atom
            atom_types[atom.symbol] = atidx + 1
    del atom_types

    #print("Starting claiming residue atoms")
    struct.residues.claim()

    #print("Starting adding bonds")
    # Process bonds in pmd Structure
    if bonds:
        for bond in obj.bonds_iter():
            pmd_bond = pmd.Bond(atom_map[bond[0].idx], atom_map[bond[1].idx])
            struct.bonds.append(pmd_bond)

    if box is None:
        if isinstance(obj, meta.Molecule):
            sys_temp = meta.System(obj)
            box = list(np.asarray(sys_temp.get_boundingbox())+5.0) + [90.0]*3
        else:
            if obj.box is None or obj.box.bounds is None:
                box = list(np.asarray(obj.get_boundingbox())+5.0) + [90.0]*3
            else:
                box_angle = list(obj.box.angle)
                box = list(obj.box.lengths) + box_angle
    elif len(box)==3:
        box = list(box) + [90.0]*3
    
    struct.box = box

    if parametrize:
        if not obj.parametrized:
            raise MetaError("Cannot convert a non-parametrized System to a parametrized ParmEd struct.")
        #print("Assigning FF parameters")
        _parametrize_pmd(obj, struct)

    return struct

def _parametrize_pmd(system, struct):
    """Add forcefield parameters to pmd Structure."""
    #import parmed as pmd
    import parmed.topologyobjects as pmdtopo

    atom_map = dict()
    atom_types = dict()
    for idx, pmdatom in enumerate(struct.atoms):
        atom = system.get_atom(idx+1)
        if atom.type not in atom_types:
            metaAtomType = system.params['atom_type'][atom.type]
            atidx = len(atom_types) + 1
            pmdAtomType = pmdtopo.AtomType(
                            name=atom.type, 
                            number=atidx, 
                            mass=metaAtomType.mass, 
                            atomic_number=metaAtomType.atomic, 
                            charge=metaAtomType.charge)
            pmdAtomType.set_lj_params(
                        eps=metaAtomType.epsilon, 
                        rmin=metaAtomType.sigma * 2**(1/6) / 2)
        else:
            pmdAtomType = atom_types[atom.type]
        pmdatom.type = atom.type
        pmdatom.epsilon = pmdAtomType.epsilon
        pmdatom.rmin = pmdAtomType.rmin
        atom_map[atom.idx] = pmdatom

    bond_types = dict()
    for bond in struct.bonds:
        bond_key = tuple(sorted((bond.atom1.type, bond.atom2.type)))
        if bond_key not in bond_types:
            metaBondType = system.params['bond_type'][bond_key]
            pmdBondType = pmdtopo.BondType(k=metaBondType.k,
                                            req=metaBondType.req)
            bond_types[bond_key] = pmdBondType
            struct.bond_types.append(pmdBondType)
        else:
            pmdBondType = bond_types[bond_key]
        bond.type = pmdBondType
    del bond_types

    angle_types = dict()
    for angle in system.angles:
        angle_key = (angle.atom1.type, angle.atom2.type, angle.atom3.type)
        if angle_key not in angle_types:
            metaAngleType = system.params['angle_type'][angle_key]
            pmdAngleType = pmdtopo.AngleType(k=metaAngleType.k,
                                            theteq=metaAngleType.theteq)
            angle_types[angle_key] = pmdAngleType
            struct.angle_types.append(pmdAngleType)
        else:
            pmdAngleType = angle_types[angle_key]
        pmdAngle = pmdtopo.Angle(
                                atom1=atom_map[angle.atom1.idx],
                                atom2=atom_map[angle.atom2.idx],
                                atom3=atom_map[angle.atom3.idx],
                                type=pmdAngleType,
                        )
        struct.angles.append(pmdAngle)
    del angle_types

    #!TODO: Add UB types

    dih_types = dict()
    for dih in system.dihedrals:
        dih_key = (dih.atom1.type, dih.atom2.type, dih.atom3.type, dih.atom4.type)
        if dih_key not in dih_types:
            metaDihType = system.params['dihedral_type'][dih_key]
            pmdDihType = pmdtopo.DihedralType(phi_k=metaDihType.phi_k,
                                            per=metaDihType.per,
                                            phase=metaDihType.phase)
            dih_types[dih_key] = pmdDihType
            struct.dihedral_types.append(pmdDihType)
        else:
            pmdDihType = dih_types[dih_key]
        pmdDihedral = pmdtopo.Dihedral(
                                atom1=atom_map[dih.atom1.idx],
                                atom2=atom_map[dih.atom2.idx],
                                atom3=atom_map[dih.atom3.idx],
                                atom4=atom_map[dih.atom4.idx],
                                type=pmdDihType,
                        )
        struct.dihedrals.append(pmdDihedral)
    del dih_types

    rb_types = dict()
    for rb_torsion in system.rb_torsions:
        rb_key = (rb_torsion.atom1.type, rb_torsion.atom2.type, rb_torsion.atom3.type, rb_torsion.atom4.type)
        if rb_key not in rb_types:
            metaRBType = system.params['rb_torsion_type'][rb_key]
            pmdRBType = pmdtopo.RBTorsionType(c0=metaRBType.c0, c1=metaRBType.c1,
                                            c2=metaRBType.c2, c3=metaRBType.c3,
                                            c4=metaRBType.c4, c5=metaRBType.c5)
            rb_types[rb_key] = pmdRBType
            struct.rb_torsion_types.append(pmdRBType)
        else:
            pmdRBType = rb_types[rb_key]
        pmdRBTorsion = pmdtopo.Dihedral(
                                atom1=atom_map[rb_torsion.atom1.idx],
                                atom2=atom_map[rb_torsion.atom2.idx],
                                atom3=atom_map[rb_torsion.atom3.idx],
                                atom4=atom_map[rb_torsion.atom4.idx],
                                type=pmdRBType,
                        )
        struct.rb_torsions.append(pmdRBTorsion)
    del rb_types

    imp_types = dict()
    for improper in system.impropers:
        imp_key = (improper.atom1.type, improper.atom2.type, improper.atom3.type, improper.atom4.type)
        if imp_key not in imp_types:
            metaImpType = system.params['improper_type'][imp_key]
            pmdImpType = pmdtopo.ImproperType(psi_k=metaImpType.psi_k, 
                                            psi_eq=metaImpType.psi_eq,
                                        )
            imp_types[imp_key] = pmdImpType
            struct.improper_types.append(pmdImpType)
        else:
            pmdImpType = imp_types[imp_key]
        pmdImproper = pmdtopo.Improper(
                                atom1=atom_map[improper.atom1.idx],
                                atom2=atom_map[improper.atom2.idx],
                                atom3=atom_map[improper.atom3.idx],
                                atom4=atom_map[improper.atom4.idx],
                                type=pmdImpType,
                        )
        struct.impropers.append(pmdImproper)
    del imp_types

def convert_to_openmm(obj, createSystem=False, forcefield_files=None, **kwargs):
    """Return openmm topology, positions and/or system."""
    if not createSystem:
        struct = convert_to_pmd(obj=obj)
        return struct.topology, struct.positions
    else:
        if forcefield_files is None:
            if not obj.parametrized:
                raise MetaError("Forcefield files are required to create an openmm system from a non-parametrized MetaSystem")
            struct = convert_to_pmd(obj=obj, parametrize=True)
            system = struct.createSystem(**kwargs)
        else:
            struct = convert_to_pmd(obj=obj)
            import openmm as mm
            forcefield = mm.app.ForceField(*forcefield_files)
            system = forcefield.createSystem(struct.topology, **kwargs)
        _cleanup_dihedrals(system)
        return struct.topology, struct.positions, system

def gromacs_to_openmm(filename, xyz=None, createSystem=False, forcefield_files=None, **kwargs):
    """Directly convert gromacs top/gro format to openmm topology, positions and/or system."""
    from metamol.utils.help_functions import parmed_load

    if filename.endswith('gro'):
        struct = parmed_load(filename)
    elif filename.endswith('top'):
        struct = parmed_load(filename, xyz=xyz)
    else:
        raise MetaError("File {0} is not a valid gromacs file".format(filename))
    
    if not createSystem:
        return struct.topology, struct.positions

    if len(struct.bonds) > 0 and struct.bonds[0].type is not None:
        parametrized = True
    else:
        parametrized = False

    if forcefield_files is None:
        if not parametrized:
            raise MetaError("Forcefield files are required to create an openmm system from a gro file only")
        system = struct.createSystem(**kwargs)
    else:
        import openmm as mm
        forcefield = mm.app.ForceField(*forcefield_files)
        system = forcefield.createSystem(struct.topology, **kwargs)        

    _cleanup_dihedrals(system)
    return struct.topology, struct.positions, system        

def _cleanup_dihedrals(system):
    """Clean up dihedral parameters for openmm system after v7.7. 
       Harmonic torsion parameters with periocity < 1 is no longer allowed."""

    import openmm as mm
    from copy import deepcopy

    TorsionForce = None
    for idx, force in enumerate(system.getForces()):
        if isinstance(force, mm.PeriodicTorsionForce):
            TorsionForce = force
            break
    if TorsionForce is None:
        return system

    for j in range(TorsionForce.getNumTorsions()):
        params = TorsionForce.getTorsionParameters(j)
        if params[-3] < 1:
            assert params[-2]._value == 0.0
            params[-3] = 1
            TorsionForce.setTorsionParameters(j, *params)

    force = deepcopy(TorsionForce)
    system.addForce(force)
    system.removeForce(idx)
    return system

def RB_to_OPLS(
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    error_tolerance=1e-4,
    error_if_outside_tolerance=True,
):
    r"""Convert Ryckaert-Bellemans type dihedrals to OPLS type.
    Inherited from mbuild.utils.conversion. Authors: Janos Sallai and Christoph Klein.

    .. math::
    RB_{torsions} &= c_0 + c_1*cos(psi) + c_2*cos(psi)^2 + c_3*cos(psi)^3 + \\
                  &= c_4*cos(psi)^4 + c_5*cos(psi)^5

    .. math::
    OPLS_torsions &= \frac{f_0}{2} + \frac{f_1}{2}*(1+cos(t)) + \frac{f_2}{2}*(1-cos(2*t)) + \\
                  &= \frac{f_3}{2}*(1+cos(3*t)) + \frac{f_4}{2}*(1-cos(4*t))

    where :math:`psi = t - pi = t - 180 degrees`

    Parameters
    ----------
    c0, c1, c2, c3, c4, c5 : Ryckaert-Belleman coefficients (in kcal/mol)
    error_tolerance : float, default=1e-4
        The acceptable absolute tolerance between the RB to OPLS conversion.
        Any value entered is converted to an absolute value.
    error_if_outside_tolerance : bool, default=True
        This variable determines whether to provide a ValueError if the RB to OPLS
        conversion is greater than the error_tolerance term (i.e., error_if_outside_tolerance=True),
        or a warning if the RB to OPLS conversion is greater than the error_tolerance term
        (i.e., error_if_outside_tolerance=False).

    Returns
    -------
    opls_coeffs : np.array, shape=(5,)
        Array containing the OPLS dihedrals coeffs f0, f1, f2, f3, and f4
        (in kcal/mol).


    Notes
    -----
    c5 must equal zero, or this conversion is not possible.

    (c0 + c1 + c2 + c3 + c4 + c5) must equal zero, to have the exact
    energies represented in the dihedral.

    NOTE: fO IS TYPICALLY NOT IN THE OPLS DIHEDRAL EQUATION AND IS
    ONLY USED TO TEST IF THIS FUNCTION CAN BE UTILIZED, AND
    DETERMINE THE fO VALUE FOR LATER ENERGY SCALING IN MOLECULAR DYNAMICS
    (MD) SIMULATIONS. THIS FUNCTION TESTS IF f0 IS ZERO (f0=0).

    .. warning:: The :math:`\frac{f_{0}}{2}` term is the constant for the OPLS dihedral equation.
        If the f0 term is not zero, the dihedral is not an exact conversion;
        since this constant does not contribute to the force equation,
        this should provide matching results for MD, but the energy for each
        dihedral will be shifted by the :math:`\frac{f_{0}}{2}` value.
    """
    if not isinstance(error_tolerance, float):
        raise TypeError(
            f"The error_tolerance variable must be a float, is type {type(error_tolerance)}."
        )
    error_tolerance = abs(error_tolerance)

    if not isinstance(error_if_outside_tolerance, bool):
        raise TypeError(
            f"The text_for_error_tolerance variable must be a bool, is type {type(error_if_outside_tolerance)}."
        )

    if not np.all(np.isclose(c5, 0, atol=1e-10, rtol=0)):
        raise ValueError(
            "c5 must equal zero, so this conversion is not possible."
        )

    f0 = 2.0 * (c0 + c1 + c2 + c3 + c4 + c5)
    if not np.all(np.isclose(f0 / 2, 0, atol=error_tolerance, rtol=0)):
        text_for_error_tolerance = (
            "f0 = 2 * ( c0 + c1 + c2 + c3 + c4 + c5 ) is not zero. "
            "The f0/2 term is the constant for the OPLS dihedral. "
            "Since the f0 term is not zero, the dihedral is not an "
            "exact conversion; since this constant does not contribute "
            "to the force equation, this should provide matching results "
            "for MD, but the energy for each dihedral will be shifted "
            "by the f0/2 value."
        )
        if error_if_outside_tolerance is True:
            raise ValueError(text_for_error_tolerance)
        elif error_if_outside_tolerance is False:
            warnings.warn(text_for_error_tolerance)

    f1 = -2 * c1 - (3 * c3) / 2
    f2 = -c2 - c4
    f3 = -c3 / 2
    f4 = -c4 / 4
    return np.asarray([f0, f1, f2, f3, f4])

def OPLS_to_RB(f0, f1, f2, f3, f4, error_tolerance=1e-4):
    r"""Convert OPLS type to Ryckaert-Bellemans type dihedrals.
    Inherited from mbuild.utils.conversion. Authors: Janos Sallai and Christoph Klein.

    .. math::
    OPLS_torsions &= \frac{f_0}{2} + \frac{f_1}{2}*(1+cos(t)) + \frac{f_2}{2}*(1-cos(2*t)) + \\
                  &= \frac{f_3}{2}*(1+cos(3*t)) + \frac{f_4}{2}*(1-cos(4*t))

    .. math::
    RB_{torsions} &= c_0 + c_1*cos(psi) + c_2*cos(psi)^2 + c_3*cos(psi)^3 + \\
                  &= c_4*cos(psi)^4 + c_5*cos(psi)^5

    where :math:`psi = t - pi = t - 180 degrees`

    Parameters
    ----------
    f0, f1, f2, f3, f4 : OPLS dihedrals coeffs (in kcal/mol)
    error_tolerance : float, default=1e-4
        The acceptable absolute tolerance between the OPLS to RB conversion
        without throwing a warning. Any value entered is converted to an
        absolute value.

    Returns
    -------
    RB_coeffs : np.array, shape=(6,)
        Array containing the Ryckaert-Bellemans dihedrals
        coeffs c0, c1, c2, c3, c4, and c5 (in kcal/mol)

    Notes
    -----
    NOTE: fO IS TYPICALLY NOT IN THE OPLS DIHEDRAL EQUATION (i.e., f0=0).

    .. warning:: The :math:`\frac{f_{0}}{2}` term is the constant for the OPLS dihedral equation,
        which is and added to a constant for the RB torsions equation via the c0 coefficient.
        If the f0 term is zero in the OPLS dihedral form or is force set to zero in this equation,
        the dihedral is may not an exact conversion;
        since this constant does not contribute to the force equation,
        this should provide matching results for MD, but the energy for each
        dihedral will be shifted by the real :math:`\frac{f_{0}}{2}` value.
    """
    if not isinstance(error_tolerance, float):
        raise TypeError(
            f"The error_tolerance variable must be a float, is type {type(error_tolerance)}."
        )
    error_tolerance = abs(error_tolerance)

    if np.all(np.isclose(f0 / 2, 0, atol=error_tolerance, rtol=0)):
        warnings.warn(
            "The f0/2 term is the constant for the OPLS dihedral equation, "
            "which is added to a constant for the RB torsions equation via the c0 coefficient. "
            "The f0 term is zero in the OPLS dihedral form or is force set to zero in this equation, "
            "so the dihedral is may not an exact conversion; "
            "since this constant does not contribute to the force equation, "
            "this should provide matching results for MD, but the energy for each"
            "dihedral will be shifted by the real f0/2 value."
        )

    c0 = f2 + (f0 + f1 + f3) / 2
    c1 = (-f1 + 3 * f3) / 2
    c2 = -f2 + 4 * f4
    c3 = -2 * f3
    c4 = -4 * f4
    c5 = 0
    return np.asarray([c0, c1, c2, c3, c4, c5])

def vectors_to_box(vectors):
    """Convert vector to box."""

    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)

    a_vec = vectors[0, :]
    b_vec = vectors[1, :]
    c_vec = vectors[2, :]

    a, b, c = np.linalg.norm(a_vec), np.linalg.norm(b_vec), np.linalg.norm(c_vec)

    cos_alpha = np.dot(b_vec, c_vec) / b / c
    cos_beta = np.dot(a_vec, c_vec) / a / c
    cos_gamma = np.dot(a_vec, b_vec) / a / b

    alpha = np.rad2deg(np.arccos(cos_alpha))
    beta = np.rad2deg(np.arccos(cos_beta))
    gamma = np.rad2deg(np.arccos(cos_gamma))

    return np.asarray((a, b, c)), np.asarray((alpha, beta, gamma))

def box_to_vectors(box, box_angle, norm=False):
    """Convert box to vector."""
    (a, b, c) = box
    (alpha, beta, gamma) = box_angle
    alpha, beta, gamma = np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)

    sin_a, cos_a = np.clip(np.sin(alpha), -1.0, 1.0), np.clip(np.cos(alpha), -1.0, 1.0)
    sin_b, cos_b = np.clip(np.sin(beta), -1.0, 1.0), np.clip(np.cos(beta), -1.0, 1.0)
    sin_g, cos_g = np.clip(np.sin(gamma), -1.0, 1.0), np.clip(np.cos(gamma), -1.0, 1.0)

    a_vec = np.asarray([1, 0.0, 0.0])

    b_x, b_y = cos_g, sin_g
    b_vec = np.asarray([b_x, b_y, 0.0])

    c_x = cos_b
    c_cos_y_term = (cos_a - cos_b * cos_g) / sin_g
    c_y = c_cos_y_term
    c_z = np.sqrt(1 - cos_b*cos_b - c_cos_y_term*c_cos_y_term)
    c_vec = np.asarray([c_x, c_y, c_z])

    if norm: 
        box_vecs = np.asarray([a_vec, b_vec, c_vec]).reshape(3, 3)
    else:
        box_vecs = np.asarray([a_vec*a, b_vec*b, c_vec*c]).reshape(3, 3)

    det = np.linalg.det(box_vecs)
    if np.isclose(det, 0.0, atol=1e-5):
        raise MetaError(
            "The vectors to define the box are co-linear, this does not form a "
            f"3D region in space.\n Box vectors evaluated: {box_vecs}"
        )
    if det < 0.0:
        warnings.warn(
            "Box vectors provided for a left-handed basis, these will be "
            "transformed into a right-handed basis automatically."
        )
        return normalize_vectors(box_vecs)
    
    return box_vecs

def normalize_vectors(vectors):
    #Align the matrix to a right-handed coordinate frame
    det = np.linalg.det(vectors)
    if np.isclose(det, 0.0, atol=1e-5):
        raise MetaError(
            "The vectors to define the box are co-linear, this does not form a "
            f"3D region in space.\n Box vectors evaluated: {vectors}"
        )
    if det < 0.0:
        warnings.warn(
            "Box vectors provided for a left-handed basis, these will be "
            "transformed into a right-handed basis automatically."
        )

    # transpose to column-major for the time being
    Q, R = np.linalg.qr(vectors.T)

    # left or right handed: det<0 left, >0, right
    sign = np.linalg.det(Q)
    R = R * sign

    signs = np.diag(
        np.diag(np.where(R < 0, -np.ones(R.shape), np.ones(R.shape)))
    )
    transformed_vecs = R.dot(signs)
    return reduced_from_vectors(transformed_vecs.T)

def reduced_from_vectors(vectors):
    #Get reduced vectors from vectors.
    a_vec = vectors[0, :]
    b_vec = vectors[1, :]
    c_vec = vectors[2, :]

    Lx = np.linalg.norm(a_vec)
    a_2x = np.dot(a_vec, b_vec) / Lx
    Ly = np.sqrt(np.dot(b_vec, b_vec) - a_2x*a_2x)
    xy = a_2x / Lx
    a_x_b = np.cross(a_vec, b_vec)
    Lz = np.dot(c_vec, (a_x_b / np.linalg.norm(a_x_b)))
    a_3x = np.dot(a_vec, c_vec) / Lx
    xz = a_3x / Lz
    yz = (np.dot(b_vec, c_vec) - a_2x * a_3x) / Lz / Ly

    reduced_vecs = np.asarray(
        [[Lx, 0.0, 0.0], [xy*Ly, Ly, 0.0], [xz*Lz, yz*Lz, Lz]])
    
    return reduced_vecs
