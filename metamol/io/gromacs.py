import os
import numpy as np
from foyer import Forcefield
from pkg_resources import resource_filename
from distutils.spawn import find_executable

import metamol as meta
from metamol.system import Box
from metamol.exceptions import GromacsError, MetaError
from metamol.utils.convert_formats import OPLS_to_RB, convert_from_pmd
from metamol.utils.help_functions import parmed_load
from metamol.utils.constants import KCAL_TO_J
from metamol.utils.ffobjects import *
from metamol.utils.help_functions import distance

__all__ = ["read_gromacs", "read_gromacs_direct", "_generate_pairs", "_read_data", "write_gromacs"]

EP_FACTOR = KCAL_TO_J / 1000
SIGMA_FACTOR = 0.1

def read_gromacs(filename: str, host_obj=None, backend='parmed', skip_bonds=True, xyz=None):
    """Read System from Gromacs(gro, top) files."""
    sys_out = meta.System()
    if backend==None or backend == 'parmed':
        GMX = find_executable('gmx') or find_executable('gmx_mpi') 

        if xyz is not None:
            struct = parmed_load(filename, xyz=xyz, GMX=GMX)
        else:
            struct = parmed_load(filename, skip_bonds=skip_bonds, GMX=GMX)
        convert_from_pmd(struct, host_obj=sys_out)

    elif backend == 'direct':
        read_gromacs_direct(filename, sys_out)
    else:
        raise MetaError("Backend {0} not supported".format(backend))

    if host_obj is None:
        return sys_out
    else:
        if filename.endswith('gro'):
            host_obj.copy_coords(target=sys_out, box=True)
            if not skip_bonds: host_obj.bond_graph = sys_out.bond_graph
        else:
            host_obj.copy(target=sys_out, coords=False)
        return host_obj

def read_gromacs_direct(filename: str, sys_out):
    """Directly read System from Gromacs files."""

    if filename.endswith('top'):
        molecules = dict()
        with open(filename, 'r') as f:
            current_section = None
            for line in f:
                line = line.strip()
                if not line or line[0] == ';': continue

                if line[0] == '[':
                    current_section = line[1:-1].strip()
                elif current_section == 'defauls':
                    words = line.split()
                    if len(words) < 2:
                        raise GromacsError('Too few fields in defaults section')
                    nbfunc, comb_rule, fulj, fuqq = int(words[0]), int(words[1]), float(words[3]), float(words[4])
                    if nbfunc != 1:
                        raise GromacsError('Now only support LJ nonbonded type')

                elif current_section == 'atomtypes':
                    a_type, a_num, mass, charge, sigma, epsilon = _read_data(line, current_section)
                    if a_type in sys_out.params['atom_type']:
                        raise GromacsError('Duplicate atom types found in gromacs topology file')
                    at_num = len(sys_out.params['atom_type']) + 1
                    sys_out.params['atom_type'][a_type] = AtomType(idx=at_num, atomic=a_num, mass=mass, charge=charge, sigma=sigma, epsilon=epsilon)

                elif current_section == 'moleculetype':
                    words = line.split()
                    mol_name, nrexcl = words[0], int(words[1])
                    if mol_name in molecules: 
                        raise GromacsError("Duplicate molecule names")
                    mol = meta.Molecule(name=mol_name)
                    molecules[mol.name] = [mol, 1]
                
                elif current_section == 'atoms':
                    aidx, atype, resname, sym, charge, mass = _read_data(line, current_section)
                    if atype not in sys_out.params['atom_type']:
                        raise GromacsError("atom type {0:s} not found in atomtypes section".format(atype))
                    atidx = sys_out.params['atom_type'][atype]
                    mol.atomList.append(meta.Atom(idx=aidx, symbol=sym, charge=charge, mass=mass, resname=resname, atomtype=atype, atidx=atidx))
                    mol.numAtoms += 1
                elif current_section == 'settles':
                    mol.add_bond((0, 1))
                    mol.add_bond((0, 2))
                    bond_key = tuple(sorted((mol[0].type, mol[1].type)))
                    if bond_key not in sys_out.params['bond_type']:
                        btidx = len(sys_out.params['bond_type']) + 1
                        sys_out.params['bond_type'][bond_key] = BondType(idx=btidx)
                    mol.angles.append(Angle(mol[1], mol[0], mol[2], angleidx=-1))
                    angle_key = (mol[1].type, mol[0].type, mol[2].type)
                    if angle_key not in sys_out.params['angle_type']:
                        angleidx = len(sys_out.params['angle_type']) + 1
                        #theteq, k = float(coeffs[0]), float(coeffs[1]) / EP_FACTOR / 2.0
                        sys_out.params['angle_type'][angle_key] = AngleType(idx=angleidx)

                elif current_section == 'bonds':
                    ai, aj, funct, coeffs = _read_data(line, current_section)
                    mol.add_bond((ai-1, aj-1))
                    bond_key = tuple(sorted((mol[ai-1].type, mol[aj-1].type)))
                    if bond_key not in sys_out.params['bond_type']:
                        btidx = len(sys_out.params['bond_type']) + 1
                        if funct == 1: # Harmonic bonds
                            req, k = float(coeffs[0]) / SIGMA_FACTOR, float(coeffs[1]) / EP_FACTOR * SIGMA_FACTOR * SIGMA_FACTOR / 2.0
                            sys_out.params['bond_type'][bond_key] = BondType(idx=btidx, k=k, req=req)

                        else:
                            raise GromacsError("Right now only supports bond type 1")

                elif current_section == 'angles':
                    ai, aj, ak, funct, coeffs = _read_data(line, current_section)
                    atom1, atom2, atom3 = mol[ai-1], mol[aj-1], mol[ak-1]
                    angle_key = (atom1.type, atom2.type, atom3.type)
                    if angle_key not in sys_out.params['angle_type']:
                        angleidx = len(sys_out.params['angle_type']) + 1
                        if funct == 1: # Harmonic angles
                            theteq, k = float(coeffs[0]), float(coeffs[1]) / EP_FACTOR / 2.0
                            sys_out.params['angle_type'][angle_key] = AngleType(idx=angleidx, k=k, theteq=theteq)
                        
                        else:
                            raise GromacsError("Right now only supports angle type 1")
                    angleidx = sys_out.params['angle_type'][angle_key].idx    
                    mol.angles.append(Angle(atom1, atom2, atom3, angleidx))
                    #sys_out.numAngles += 1

                elif current_section == 'dihedrals':
                    ai, aj, ak, al, funct, coeffs = _read_data(line, current_section)
                    atom1, atom2, atom3, atom4 = mol[ai-1], mol[aj-1], mol[ak-1], mol[al-1]
                    dih_key = (atom1.type, atom2.type, atom3.type, atom4.type)
                    if funct == 1: # Proper dihedral
                        if dih_key not in sys_out.params['dihedral_type']:
                            dihidx = len(sys_out.params['dihedral_type']) + 1
                            phase, phi_k, per = coeffs[0], coeffs[1] / EP_FACTOR / 2.0, coeffs[2]
                            sys_out.params['dihedral_type'][dih_key] = DihedralType(idx=dihidx, phi_k=phi_k, per=per, phase=phase)
                        dihidx = sys_out.params['dihedral_type'][dih_key].idx
                        mol.dihedrals.append(Dihedral(atom1, atom2, atom3, atom4, dihidx))
                        #sys_out.numDihedrals += 1
                    elif funct == 2: # Improper dihedral
                        if dih_key not in sys_out.params['improper_type']:
                            impidx = len(sys_out.params['improper_type']) + 1
                            psi_eq, psi_k = coeffs[0], coeffs[1] / EP_FACTOR /2.0
                            sys_out.params['improper_type'][dih_key] = ImproperType(idx=impidx, psi_k=psi_k, psi_eq=psi_eq)
                        impidx = sys_out.params['improper_type'][dih_key].idx
                        mol.impropers.append(Improper(atom1, atom2, atom3, atom4, itidx=impidx))
                        #sys_out.numImpropers += 1
                    elif funct == 3: # RB torsion
                        if dih_key not in sys_out.params['rb_torsion_type']:
                            rbidx = len(sys_out.params['rb_torsion_type']) + 1
                            [c0, c1, c2, c3, c4, c5] = [float(c) / EP_FACTOR for c in coeffs]
                            sys_out.params['rb_torsion_type'][dih_key] = RBTorsionType(idx=rbidx, c0=c0, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5)
                        rbidx = sys_out.params['rb_torsion_type'][dih_key].idx
                        mol.rb_torsions.append(RB_Torsion(atom1, atom2, atom3, atom4, rbidx=rbidx))
                        #sys_out.numRBs += 1
                    
                elif current_section == 'system':
                    name = line.strip()
                    if name != 'Generic title':
                        sys_out.name = name
                
                elif current_section == 'molecules':
                    words = line.split()
                    molname, numMols = words[0], int(words[1])
                    if molname not in molecules:
                        raise GromacsError("Molecule name in molecules section not found in moleculetype section")
                    molecules[molname][1] = numMols

        sys_out.parametrized = True
        for (mol, dup) in molecules.values():
            sys_out.add(mol, dup, update_index=False)
        del molecules
        sys_out.flatten()

    elif filename.endswith('gro'):
        _num_digits = None
        with open(filename, 'r') as f:
            f.readline()
            numAtoms = int(f.readline().strip())
            molid = 0
            for k in range(numAtoms):
                line = f.readline()
                resnum = int(line[:5])
                resname = line[5:10].strip()
                atomname = line[10:15].strip()
                atomidx = int(line[15:20])
                if _num_digits is None:
                    _first_deci_index = line.index('.', 20)
                    _second_deci_index = line.index('.', _first_deci_index+1)
                    _num_digits = _second_deci_index - _first_deci_index
                x, y, z = (
                    float(line[20+i*_num_digits:20+(i+1)*_num_digits]) / SIGMA_FACTOR for i in range(3)
                )
                atom = meta.Atom(idx=atomidx, symbol=atomname, resname=resname, resid=resnum, x=x, y=y, z=z)

                if resnum != molid:
                    if 'mol' in locals():
                        sys_out.add(mol, update_index=False)
                    mol = meta.Molecule(atom, name=resname)
                    molid += 1
                else:
                    mol.atomList.append(atom)
                    mol.numAtoms += 1

                if k == numAtoms - 1:
                    sys_out.add(mol, update_index=False)

                box_read = f.readline().split()
                box_bounds = [0.0, 0.0, 0.0] + [float(box_read[idx]) / SIGMA_FACTOR for idx in range(3)]
                box = Box(bounds=box_bounds)
                sys_out.box = box

    else:
        raise MetaError("File {0} is not recognized as a gromacs file".format(filename))

def _generate_pairs(mol, n):
    """Generate the n-th pairs in the input Molecule."""
    pairs = set()
    start_idx = mol.atoms[0].idx
    for atom in mol.atoms:
        seen = set([atom.idx])
        level = [atom]
        count = 0
        while level and count < n-1:
            #print(atom.idx, count, len(level))
            temp = []
            for node in level:
                for neigh in mol.bond_graph.neighbors_iter(node):
                    if neigh.idx not in seen:
                        seen.add(neigh.idx)
                        temp.append(neigh)
            level = temp
            count += 1
        for node in level:
            pair = tuple(sorted((atom.idx-start_idx+1, node.idx-start_idx+1)))
            pairs.add(pair)
    return list(pairs)

def _read_data(line, section):
    """Read atom type information from the line"""
    words = line.split()
    if section == 'atomtypes':
        if len(words) < 7:
            raise GromacsError("Too few fields in section {0:s}".format(section))
        return words[0], int(words[1]), float(words[2]), float(words[3]), float(words[5])/SIGMA_FACTOR, float(words[6])/EP_FACTOR
    if section == 'atoms':
        if len(words) < 8:
            raise GromacsError("Too few fields in section {0:s}".format(section))
        return int(words[0]), words[1], words[3], words[4], float(words[6]), float(words[7])
    if section == 'bonds':
        if len(words) < 3:
            raise GromacsError("Too few fields in section {0:s}".format(section))
        return int(words[0]), int(words[1]), int(words[2]), words[3:]
    if section == 'angles':
        if len(words) < 4:
            raise GromacsError("Too few fields in section {0:s}".format(section))
        return int(words[0]), int(words[1]), int(words[2]), int(words[3]), words[4:]
    if section == 'dihedrals':
        if len(words) < 5:
            raise GromacsError("Too few fields in section {0:s}".format(section))
        return int(words[0]), int(words[1]), int(words[2]), int(words[3]), int(words[4]), words[5:]

def write_gromacs(sys, filename: str, **kwargs):
    """Write the System object to Gromacs(gro, top) files.
    
    Parameters
    ----------
    sys : metamol.System
        The model system to write.
    filename : str
        Name of the file to write to.

    Optional Parameters
    ----------
    forcefield_files : str, default=None
        The location of forcefield files to parametrize the system. Invoked when the input System is not parametrized.
    forcefield_name : str, default=opls
        The name of forcefield to parametrize the system. Invoked when the input System is not parametrized and foecefield_files is None.

    combining_rule : int, default=3
        The rule used to combine pair coefficients. 1: geometric for C12, C6, 2: Lorentz-Berthelot, 3: geometric for both sigma and epsilon.
    nrexcl : int, default=3
        How many neighbors to exclude from non-bonded interactins. For example, 3 means excluding 1-4 neighbors.
    nb_funct : int, default=1
        Nonbonded interaction type. 1: LJ. 2: Buckingham.
    fudgeLJ : float, default=1.0
        The factor by which to multiply Lennard-Jones 1-4 interactions.
    fudgeQQ : float, default=1.0
        The factor by which to multiply electrostatic 1-4 interactions.
    b_funct : int, default=1
        The bond interaction type. 1: bond. 2: G96 bond (to add). 3: Morse (to add).
    a_funct : int, default=1
        The angle interaction type. 1: angle. 2: G96 angle (to add). 5: Urey-Bradley (to add).
    pairs: bool, default=False
        Whether to add a [ pairs ] section in the topology file.
    opls: bool, default=False
        Whether the opls force field is used.
    n_neighbor: int, default=4
        Degree of neighbors to include in the [ pairs ] section.

    Outputs
    -------
    filename : Gromacs gro/top files    
    """
    
    if filename and not (filename.endswith('gro') or filename.endswith('top')):
        write_gromacs(sys, filename+'.gro', **kwargs)
        write_gromacs(sys, filename+'.top', **kwargs)

    # Write gro file.
    elif filename.endswith('gro'):
        ndeci = kwargs.get('decimal', 5)
        pdeci, ndeci = str(ndeci+5), str(ndeci)
        pre = pdeci+'.'+ndeci
        with open(filename, 'w') as f:
            f.write("GROningen MAchine for Chemical Simulation (Created by MetaMol (version={0}))\n".format(meta.__version__))
            f.write(" {0:d}\n".format(int(sys.numAtoms)))
            atom_line = '{resid:5d}{res:5s}{atomty:>5s}{atidx:5d}{x:'+pre+'f}{y:'+pre+'f}{z:'+pre+'f}\n'
            for molid, mol in enumerate(sys.molecules):
                for atidx, atom in enumerate(mol.atoms):
                    res = 'RES' if not atom.resname else atom.resname[:5]
                    resid = molid+1 if atom.resid==-1 else atom.resid
                    atomty = atom.name[:5] if atom.name else atom.symbol
                    f.write(
                    atom_line.format(
                        resid=resid,
                        res=res,
                        atomty=atomty,
                        atidx=atom.idx,
                        x = atom.x * SIGMA_FACTOR,
                        y = atom.y * SIGMA_FACTOR,
                        z = atom.z * SIGMA_FACTOR,
                    )
                )
            box_line = '\t{0:'+pre+'f}\t{1:'+pre+'f}\t{2:'+pre+'f}\n'
            f.write(box_line.format(sys.box.lengths[0] * SIGMA_FACTOR, sys.box.lengths[1] * SIGMA_FACTOR, sys.box.lengths[2] * SIGMA_FACTOR))

    # Write top file.
    elif filename.endswith('top'):
        if not sys.parametrized:
            forcefield_files = kwargs.get('forcefield_files', None)
            forcefield_name = kwargs.get('forcefield_name', 'opls')
            if forcefield_name:
                ff_file = resource_filename("metamol", os.path.join("ff_files", forcefield_name.lower()+".xml"))

                if forcefield_files is None:
                    forcefield_files = ff_file

            ff = Forcefield(forcefield_files=forcefield_files)
            struct = sys.to_pmd()
            struct = ff.apply(struct)
            struct.save(filename, overwrite=True)
            return

        cr = kwargs.get('combining_rule', 3)
        nrexcl = kwargs.get('nrexcl', 3)
        nb_funct = kwargs.get('nb_funct', 1)
        b_funct = kwargs.get('b_funct', 1)
        a_funct = kwargs.get('a_funct', 1)
        fulj = kwargs.get('fudgeLJ', 1.0)
        fuqq = kwargs.get('fudgeQQ', 1.0)
        include_pair = kwargs.get('pairs', False)

        opls = kwargs.get('opls', False)
        if 'opls' in sys.ff_name:
            opls = True

        # If opls force filed is used, need to set the 1-4 interaction scale to 0.5
        if opls:
            fulj, fuqq = 0.5, 0.5
            include_pair = True

        with open(filename, 'w') as f:
            f.write(";\n")
            f.write(";\t GROningen MAchine for Chemical Simulation topology file\n")
            f.write(";\t Created by MetaMol (version={0})\n".format(meta.__version__))
            f.write(";\n\n")
            f.write("[ defaults ]\n")
            f.write("; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ\n")
            f.write("{0:8d}{1:17d}{2:>16s}{3:14.3f}{4:8.3f}\n".format(nb_funct, cr, "yes", fulj, fuqq))
            f.write("\n[ atomtypes ]\n")
            f.write(";name	at.num	mass	charge	ptype	sigma	epsilon\n")
            atline = "{name:s}\t{atnum:s}\t{mass:8.6f}\t{charge:.6f}\t{pt:s}\t{sigma:.12f}\t{epsilon:.6f}\n"
            for key, at in sys.params['atom_type'].items():
                f.write(
                    atline.format(
                        name=key,
                        atnum=str(at.atomic),
                        mass=at.mass,
                        charge=at.charge,
                        pt="A",
                        sigma=at.sigma * SIGMA_FACTOR,
                        epsilon=at.epsilon * EP_FACTOR,
                        )
                    )
            f.write("\n")
            moldict = dict()
            for molnum, mol in enumerate(sys.molecules):
                name = mol.name
                if not name:
                    name = 'MOL' + str(molnum+1)
                if name in moldict:
                    moldict[name] += 1
                else:
                    moldict[name] = 1
                    f.write("\n[ moleculetype ]\n")
                    f.write("; Name\t nrexcl\n")
                    f.write("{0:5s}\t{1:4d}\n".format(name, nrexcl))
                    f.write("\n[ atoms ]\n")
                    f.write(";   nr       type  resnr residue  atom   cgnr     charge       mass\n")
                    atomline = "{nr:6d}{type:>11s}{resnr:7d}{residue:>7s}{atsym:>7s}{cgnr:7d}{charge:11.6f}{mass:11.6f}   ;\n"
                    for idx, atom in enumerate(mol.atoms):
                        atsym = atom.name if atom.name else atom.symbol
                        res = 'RES' if not atom.resname else atom.resname[:5]
                        f.write(
                            atomline.format(
                                nr=idx+1,
                                type=atom.type,
                                resnr=atom.resid,
                                residue=res,
                                atsym=atsym,
                                cgnr=1,
                                charge=atom.charge,
                                mass=atom.mass,
                                )
                            )

                    if 'water' in mol.name.lower():
                        constraint = 'settles'
                        O, H1, H2 = mol[0], mol[1], mol[2]
                        doh = sys.params['bond_type'][tuple(sorted((O.type, H1.type)))].req * SIGMA_FACTOR
                        dhh = distance(H1.xyz, H2.xyz) * SIGMA_FACTOR

                        f.write("\n[ {} ]\n".format(constraint))
                        f.write("; OW\tfunct\tdoh\tdhh\n")
                        f.write("{0:d}\t{1:d}\t{2:f}\t{3:f}\n".format(1, 1, doh, dhh))
                           
                        f.write("\n[ exclusions ]\n")                            
                        exline = "{0:d}\t{1:d}\t{2:d}\n"
                        f.write(exline.format(1, 2, 3))
                        f.write(exline.format(2, 1, 3))
                        f.write(exline.format(3, 1, 2))

                    else:
                        start_idx, end_idx = mol.atoms[0].idx, mol.atoms[-1].idx

                        f.write("\n[ bonds ]\n")
                        f.write(";    ai     aj funct         c0         c1         c2         c3\n")
                        bondline = "{ai:7d}{aj:7d}{funct:6d}{b:11.6f}{kb:14.6f}\n"
                        for bond in mol.bonds_iter():
                            ai, aj = bond[0], bond[1]
                            bond_key = tuple(sorted((ai.type, aj.type)))
                            bt = sys.params['bond_type'][bond_key]
                            f.write(
                                bondline.format(
                                    ai=ai.idx-start_idx+1,
                                    aj=aj.idx-start_idx+1,
                                    funct=b_funct,
                                    b=bt.req * SIGMA_FACTOR,
                                    kb=bt.k * 2 * EP_FACTOR / SIGMA_FACTOR / SIGMA_FACTOR,
                                    )
                                )


                        if include_pair:
                            n_neigh = kwargs.get('n_neighbor', 4)
                            pairs = _generate_pairs(mol, n_neigh)

                            if pairs:
                                f.write("\n[ pairs ]\n")
                                f.write(";    ai     aj funct         c0         c1         c2         c3\n")
                                pairline = "{ai:7d}{aj:7d}{funct:6d}\n"
                                for pair in pairs:
                                    f.write(
                                        pairline.format(
                                            ai=pair[0],
                                            aj=pair[1],
                                            funct=nb_funct,
                                            )
                                        )

                        angle_mol = []
                        for angle in sys.angles:
                            if start_idx <= angle.atom1.idx <= end_idx:
                                ang_key = (angle.atom1.type, angle.atom2.type, angle.atom3.type)
                                at = sys.params['angle_type'][ang_key]
                                angle_mol.append((angle.atom1.idx-start_idx+1, angle.atom2.idx-start_idx+1, angle.atom3.idx-start_idx+1, a_funct, at.theteq, at.k))
                            else:
                                continue

                        f.write("\n[ angles ]\n")
                        f.write(";    ai     aj     ak funct         c0         c1         c2         c3\n")
                        angline = "{ai:7d}{aj:7d}{ak:7d}{funct:6d}{theta:11.6f}{kt:11.6f}\n"
                        for angle in angle_mol:
                            f.write(
                                angline.format(
                                    ai=angle[0],
                                    aj=angle[1],
                                    ak=angle[2],
                                    funct=angle[3],
                                    theta=angle[4],
                                    kt=angle[5] * 2 * EP_FACTOR,
                                    )
                                )                                                    
                        del angle_mol

                        dih_mol = []
                        for dih in sys.dihedrals:
                            if start_idx <= dih.atom1.idx <= end_idx:
                                dih_key = (dih.atom1.type, dih.atom2.type, dih.atom3.type, dih.atom4.type)
                                dt = sys.params['dihedral_type'][dih_key]
                                dih_mol.append((dih.atom1.idx-start_idx+1, dih.atom2.idx-start_idx+1, dih.atom3.idx-start_idx+1, dih.atom4.idx-start_idx+1, 1, int(dt.phase), dt.phi_k, dt.per))
                            else:
                                continue

                        if dih_mol:
                            f.write("\n[ dihedrals ]\n")
                            f.write(";    ai     aj     ak funct         c0         c1         c2         c3\n")
                            dihline = "{ai:7d}{aj:7d}{ak:7d}{al:7d}{funct:6d}{phase:11d}{phi_k:11.6f}{per:11d}\n"
                            for dih in dih_mol:
                                f.write(
                                    dihline.format(
                                        ai=dih[0],
                                        aj=dih[1],
                                        ak=dih[2],
                                        al=dih[3],
                                        funct=dih[4],
                                        phase=dih[5],
                                        phi_k=dih[6] * 2 * EP_FACTOR,
                                        per=dih[7],
                                        )
                                    )
                        else:
                            for rb in sys.rb_torsions:
                                if start_idx <= rb.atom1.idx <= end_idx:
                                    rb_key = (rb.atom1.type, rb.atom2.type, rb.atom3.type, rb.atom4.type)
                                    rbt = sys.params['rb_torsion_type'][rb_key]
                                    if rbt.opls:
                                        rb_params = OPLS_to_RB(rbt.c1, rbt.c2, rbt.c3, rbt.c4, rbt.c5)
                                    else:
                                        rb_params = (rbt.c0, rbt.c1, rbt.c2, rbt.c3, rbt.c4, rbt.c5)
                                    dih_mol.append((rb.atom1.idx-start_idx+1, rb.atom2.idx-start_idx+1, rb.atom3.idx-start_idx+1, rb.atom4.idx-start_idx+1, 3, 
                                                    rb_params[0], rb_params[1], rb_params[2], rb_params[3], rb_params[4], rb_params[5]))
                                else:
                                    continue
 
                            if dih_mol: 
                                f.write("\n[ dihedrals ]\n")
                                f.write(";    ai     aj     ak     al funct         c0         c1         c2         c3         c4         c5\n")
                                dihline = "{ai:7d}{aj:7d}{ak:7d}{al:7d}{funct:6d}{c0:11.6f}{c1:11.6f}{c2:11.6f}{c3:11.6f}{c4:11.6f}{c5:11.6f}\n"
                                for dih in dih_mol:
                                    f.write(
                                        dihline.format(
                                            ai=dih[0],
                                            aj=dih[1],
                                            ak=dih[2],
                                            al=dih[3],
                                            funct=dih[4],
                                            c0=dih[5] * EP_FACTOR,
                                            c1=dih[6] * EP_FACTOR,
                                            c2=dih[7] * EP_FACTOR,
                                            c3=dih[8] * EP_FACTOR,
                                            c4=dih[9] * EP_FACTOR,
                                            c5=dih[10] * EP_FACTOR,
                                            )
                                        ) 
                        del dih_mol                                                                                                                                                               

                        imp_mol = []
                        for imp in sys.impropers:
                            if start_idx <= imp.atom1.idx <= end_idx:
                                imp_key = (imp.atom1.type, imp.atom2.type, imp.atom3.type, imp.atom4.type)
                                imt = sys.params['improper_type'][imp_key]
                                imp_mol.append((imp.atom1.idx-start_idx+1, imp.atom2.idx-start_idx+1, imp.atom3.idx-start_idx+1, imp.atom4.idx-start_idx+1, 2, imt.psi_eq, imt.psi_k))
                            else:
                                continue

                        if imp_mol:
                            f.write("\n[ dihedrals ]\n")
                            f.write(";    ai     aj     ak     al funct         c0         c1         c2         c3\n")
                            impline = "{ai:7d}{aj:7d}{ak:7d}{al:7d}{funct:6d}{psi_eq:11.6f}{psi_k:11.6f}\n"
                            for imp in imp_mol:
                                f.write(
                                    impline.format(
                                        ai=imp[0],
                                        aj=imp[1],
                                        ak=imp[2],
                                        al=imp[3],
                                        funct=imp[4],
                                        psi_eq=imp[5],
                                        psi_k=imp[6] * 2 * EP_FACTOR,
                                        )
                                    )
                        del imp_mol

            f.write("\n[ system ]\n")
            f.write("; Name\n")
            if sys.name:
                f.write("{0}\n".format(sys.name))
            else:
                f.write("Generic title\n")
            
            f.write("\n[ molecules ]\n")
            f.write("; Compound \t #mols\n")
            for key in moldict.keys():
                f.write("{0} \t {1:d}\n".format(key, moldict[key]))
            f.write("\n")

                    