import pytest

from metamol.lib.molecules.alkane import *
from metamol.lib.molecules.polymer import *
from metamol.lib.fragments.monomers import *

def __get_param__(monomers_seq=[], head=[], tail=[], N=[]):
    for m_seq in monomers_seq:
        for h in head:
            for t in tail:
                for num in N:
                    monomers = [Monomer(mol=m_seq[0][i][0], head=m_seq[0][i][1][0], tail=m_seq[0][i][1][1], remove_atoms=m_seq[0][i][2]) for i in range(len(m_seq[0]))]
                    yield (monomers, m_seq[1], h, t, num)

@pytest.mark.parametrize('monomers,seq,head,tail,N', __get_param__(
    monomers_seq = [([('CC', (1, 2), (3, 6))], 'A'), ([('CCO', (1, 3), (4, 9))], 'A'), ([('c1cc(CCCO)ccc1', (10, 7), (19, 22))], 'A'),
    ([('CC', (1, 2), (3, 6)), ('CCO', (1, 3), (4, 9))], 'AB'),
    ([('CC', (1, 2), (3, 6)), ('CCO', (1, 3), (4, 9)), ('c1cc(CCCO)ccc1', (10, 7), (19, 22))], 'ABC')],
    head = [None, CH3(), OH()], tail = [None, CH3(), OH()],
    N = [1, 5, 10],
    )
)
def test_polymer_general(monomers, seq, head, tail, N):
    pol = Polymer(monomers, seq=seq, head=head, tail=tail)
    pol.build(N)

    headAtoms = 1 if head is None else head.numAtoms
    tailAtoms = 1 if tail is None else tail.numAtoms
    assert pol.numAtoms == sum([m.numAtoms for m in monomers]) * N + headAtoms + tailAtoms

    headBonds = 0 if head is None else head.numBonds
    tailBonds = 0 if tail is None else tail.numBonds
    assert pol.numBonds == sum([m.numBonds+1 for m in monomers]) * N + headBonds + tailBonds + 1

    all_ele_frags = ['H'] if head is None else [a.symbol for a in head.atoms_iter()]
    all_ele_frags += ['H'] if tail is None else [a.symbol for a in tail.atoms_iter()]
    for monomer in monomers:
        all_ele_frags += [a.symbol for a in monomer.atoms_iter()] * N
    
    all_ele_pol = [a.symbol for a in pol.atoms_iter()]
    
    assert len(all_ele_frags) == len(all_ele_pol)

    from collections import Counter
    assert Counter(all_ele_frags) == Counter(all_ele_pol)

    return