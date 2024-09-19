import numpy as np
import os
import re
from glob import glob
from collections import Counter
from typing import List
from pathlib import Path

import pymbar

from metamol.metamol import MetaError

# Define some utils functions
def trPy(s, l='[,\\\\"/()-]', char=' '):
    """In string 's' replace all the charachters from 'l' with 'char'."""
    return re.sub(l, char, s)

def wcPy(f):
    """Count up lines in file 'f'."""
    with open(f, 'r') as fp:
        lines = len(fp.readlines())
    return lines

def tail(filepath, n, block=-1024):
    with open(filepath, 'rb') as f:
        f.seek(0,2)
        filesize = f.tell()
        while True:
            if filesize >= abs(block):
                f.seek(block, 2)
                s = f.readlines()
                if len(s) > n:
                    return s[-n:]
                    break
                else:
                    block *= 2
            else:
                block = -filesize

#TODO: Rewrite load data with pandas dataframe
def readXvgData(files: List[str],            # list of xvg file path
                temperature: float = 298.15, # temperature, in K
                equiltime: float = 0.0,      # equilibration time, in ps
            ):
    """Read in .xvg files; return nsnapshots, lv, dhdlt, and u_klt."""

    class F:
        """This is the object to be built on the filename."""

        def __init__(self, filename):
            self.filename = filename

        def readHeader(self):
            self.skip_lines = 0  # Number of lines from the top that are to be skipped.
            self.lv_names   = () # Lambda type names, e.g. 'coul', 'vdw'.
            snap_size       = [] # Time from first two snapshots to determine snapshot's size.
            self.lv         = [] # Lambda vectors, e.g. (0, 0), (0.2, 0), (0.5, 0).

            self.bEnergy    = False
            self.bPV        = False
            self.bExpanded  = False
            self.temperature= -1.0
            self.fep_state = -1

            print("Reading metadata from {0}...".format(self.filename))
            with open(self.filename,'r') as infile:
                for line in infile:
                    if line.startswith('#'):
                        self.skip_lines += 1

                    elif line.startswith('@'):
                        self.skip_lines += 1
                        elements = trPy(line).split()
                        if not 'legend' in elements:
                            if 'T' in elements:
                                self.temperature = float(elements[4])
                                self.fep_state = int(elements[9][:-1])
                            continue

                        if 'Energy' in elements:
                            self.bEnergy = True
                        if 'pV' in elements:
                            self.bPV = True
                        if 'state' in elements:
                            self.bExpanded = True

                        if 'dH' in elements:
                            self.lv_names += elements[7],
                        if 'xD' in elements:
                            self.lv.append(elements[-len(self.lv_names):])

                    else:
                        snap_size.append(float(line.split()[0]))
                        #break
                        if len(snap_size) > 1:
                           self.snap_size = np.diff(snap_size)[0]
                           break
            return self.lv

        def iter_loadtxt(self, state):
            """Houstonian Joe Kington claims it is faster than numpy.loadtxt:
            http://stackoverflow.com/questions/8956832/python-out-of-memory-on-large-csv-file-numpy"""

            def iter_func():
                with open(self.filename, 'r') as infile:
                   for _ in range(self.skip_lines):
                       next(infile)
                   for line in infile:
                      line = line.split()
                      for item in line:
                         yield item

            def slice_data(data, state=state):
                # Where the dE columns should be stored.
                if (len(ndE_unique)>1 and ndE[state]<4):
                   # If BAR, store shifted 2/3 arrays.
                   s1, s2 = np.array((0, ndE[state])) + state-(state>0)
                else:
                   # If MBAR or selective MBAR or BAR/MBAR, store all.
                   s1, s2 = (0, K)
                # Which dhdl columns are to be read.
                read_dhdl_sta = 1+self.bEnergy+self.bExpanded
                read_dhdl_end = read_dhdl_sta + n_components

                data = data.T
                dhdlt[state, :, nsnapshots_l[state]:nsnapshots_r[state]] = data[read_dhdl_sta : read_dhdl_end, :]

                if not bSelective_MBAR:
                   r1, r2 = ( read_dhdl_end, read_dhdl_end + (ndE[state] if not self.bExpanded else K) )
                   if bPV:
                      u_klt[state, s1:s2, nsnapshots_l[state]:nsnapshots_r[state]] = beta * ( data[r1:r2, :] + data[-1,:] )
                   else:
                      u_klt[state, s1:s2, nsnapshots_l[state]:nsnapshots_r[state]] = beta * data[r1:r2, :]
                else: # can't do slicing; prepare a mask (slicing is thought to be faster/less memory consuming than masking)
                   mask_read_uklt = np.array( [0]*read_dhdl_end + [1 if (k in sel_states) else 0 for k in range(ndE[0])] + ([0] if bPV else []), bool )
                   if bPV:
                      u_klt[state, s1:s2, nsnapshots_l[state]:nsnapshots_r[state]] = beta * ( data[mask_read_uklt, :] + data[-1,:] )
                   else:
                      u_klt[state, s1:s2, nsnapshots_l[state]:nsnapshots_r[state]] = beta * data[mask_read_uklt, :]
                return

            print("Loading in data from %s (%s) ..." % (self.filename, "all states" if self.bExpanded else 'state %d' % state))
            data = np.fromiter(iter_func(), dtype=float)
            if not self.len_first == self.len_last:
                data = data[: -self.len_last]
            data = data.reshape((-1, self.len_first))

            if self.bExpanded:
                for k in range(K):
                    mask_k = (data[:, 1] == k)
                    data_k = data[mask_k]
                    slice_data(data_k, k)
            else:
                slice_data(data)

    kB = 1.3806488*6.02214129/1000.0 # Boltzmann's constant (kJ/mol/K).
    beta = 1./(kB*temperature)

    fs = [ F(filename) for filename in files ]
    n_files = len(fs)

    # Preliminaries I: Sort the xvg files; read in the @-header.
    lv = []
    snap_size = []
    for nf, f in enumerate(fs):
        lv.append(f.readHeader())

        if nf>0:
            if not f.lv_names == lv_names:
                if not len(f.lv_names) == n_components:
                   raise SystemExit("\nERROR!\nFiles do not contain the same number of lambda gradient components; I cannot combine the data.")
                else:
                   raise SystemExit("\nERROR!\nThe lambda gradient components have different names; I cannot combine the data.")
            if not f.bPV == bPV:
                raise SystemExit("\nERROR!\nSome files contain the PV energies, some do not; I cannot combine the files.")
            if not f.temperature == f_temperature: # compare against a string, not a float.
                raise SystemExit("\nERROR!\nTemperature is not the same in all .xvg files.")

        else:
            lv_names = lv_names = f.lv_names

            f_temperature = f.temperature
            if f_temperature > 0:
                beta *= temperature/f_temperature
                temperature = f_temperature
                print("Temperature is %s K." % temperature)
            else:
                 print("Temperature not present in xvg files. Using %g K." % temperature)

            n_components = len(lv_names)
            bPV = f.bPV
            P_bExpanded = f.bExpanded

    # Sort fs based on state id
    fs.sort(key = lambda x: x.fep_state)

    # Preliminaries II: Analyze data for validity; build up proper 'lv' and count up lambda states 'K'.
    ndE = [len(i) for i in lv]
    ndE_unique = np.unique(ndE)

    # Scenario #1: Each file has all the dE columns -- can use MBAR.
    if len(ndE_unique) == 1: # [K]
        if not np.array([i == lv[0] for i in lv]).all():
            raise SystemExit("\nERROR!\nArrays of lambda vectors are different; I cannot combine the data.")
        else:
            lv = lv[0]
             # Handle the case when only some particular files/lambdas are given.
            if 1 < n_files < len(lv) and not P_bExpanded:
                bSelective_MBAR = True
                sel_states = [f.state for f in fs]
                lv = [lv[i] for i in sel_states]
            else:
                bSelective_MBAR = False

    elif len(ndE_unique) <= 3:
        bSelective_MBAR = False
        # Scenario #2: Have the adjacent states only; 2 dE columns for the terminal states, 3 for inner ones.
        if ndE_unique.tolist() == [2, 3]:
            lv  = [l[i>0]  for i,l in enumerate(lv)]
        # Scenario #3: Have a mixture of formats (adjacent and all): either [2,3,K], or [2,K], or [3,K].
        else:
            lv = lv[ndE_unique.argmax()]

    else:
        print("The files contain the number of the dE columns I cannot deal with; will terminate.\n\n%-10s %s " % ("# of dE's", "File"))
        for nf, f in enumerate(fs):
            print("%6d     %s" % (ndE[nf], f.filename))
        raise SystemExit("\nERROR!\nThere are more than 3 groups of files (%s, to be exact) each having different number of the dE columns; I cannot combine the data." % len(ndE_unique))

    lv = np.array(lv, float)    # Lambda vectors.
    K  = len(lv)                # Number of lambda states.

    # Preliminaries III: Count up the equilibrated snapshots.
    nsnapshots = np.zeros((n_files, K), int)
    for nf, f in enumerate(fs):
        f.len_first, f.len_last = [len(line.decode().split()) for line in tail(f.filename, 2)]
        bLenConsistency = (f.len_first != f.len_last)

        equilsnapshots  = int(equiltime/f.snap_size)
        f.skip_lines   += equilsnapshots
        nsnapshots[nf,nf] += wcPy(f.filename) - f.skip_lines - 1*bLenConsistency

        print("First %s ps (%s snapshots) will be discarded due to equilibration from file %s..." % (equiltime, equilsnapshots, f.filename))

    # Preliminaries IV: Load in equilibrated data.
    maxn  = max(nsnapshots.sum(axis=0))                       # maximum number of the equilibrated snapshots from any state
    dhdlt = np.zeros([K,n_components,int(maxn)], float)       # dhdlt[k,n,t] is the derivative of energy component n with respect to state k of snapshot t
    u_klt = np.zeros([K,K,int(maxn)], np.float64)             # u_klt[k,m,t] is the reduced potential energy of snapshot t of state k evaluated at state m

    nsnapshots = np.concatenate((np.zeros([1, K], int), nsnapshots))
    for nf, f in enumerate(fs):
        nsnapshots_l = nsnapshots[:nf+1].sum(axis=0)
        nsnapshots_r = nsnapshots[:nf+2].sum(axis=0)
        f.iter_loadtxt(nf)

    return nsnapshots.sum(axis=0), lv, dhdlt, u_klt

def uncorrelate(nsnapshots, lv, dhdlt, u_klt, sta, fin, uncorr_threshold=50, do_dhdl=False):
    """Identifies uncorrelated samples and updates the arrays of the reduced potential energy and dhdlt retaining data entries of these samples only.
      'sta' and 'fin' are the starting and final snapshot positions to be read, both are arrays of dimension K."""
    K = u_klt.shape[0]
    n_components = lv.shape[1]
    u_kln = np.zeros([K, K, max(fin-sta)], np.float64) # u_kln[k,m,n] is the reduced potential energy of uncorrelated sample index n from state k evaluated at state m
    N_k = np.zeros(K, int) # N_k[k] is the number of uncorrelated samples from state k
    g = np.zeros(K, float) # autocorrelation times for the data
    if do_dhdl:
        dhdl = np.zeros([K, n_components, max(fin-sta)], float) #dhdl is value for dhdl for each component in the file at each time.
        print("\n\nNumber of correlated and uncorrelated samples:\n\n%6s %12s %12s %12s\n" % ('State', 'N', 'N_k', 'N/N_k'))

    lchange = np.zeros([K, n_components],bool)   # booleans for which lambdas are changing
    for j in range(n_components):
      # need to identify range over which lambda doesn't change, and not interpolate over that range.
      for k in range(K-1):
         if (lv[k+1,j]-lv[k,j] > 0):
            lchange[k,j] = True
            lchange[k+1,j] = True

    # Uncorrelate based on dhdl values at a given lambda.
    for k in range(K):
        # Sum up over those energy components that are changing.
        # if there are repeats, we need to use the lchange[k] from the last repeated state.
        lastl = k
        for l in range(K):
            if np.array_equal(lv[k], lv[l]):
                lastl = l
        dhdl_sum = np.sum(dhdlt[k, lchange[lastl], sta[k]:fin[k]], axis=0)
        # Determine indices of uncorrelated samples from potential autocorrelation analysis at state k

        #NML: Set statistical inefficiency (g) = 1 if vector is all 0
        if not np.any(dhdl_sum):
            #print "WARNING: Found all zeros for Lambda={}\n Setting statistical inefficiency g=1.".format(k)
            g[k] = 1
        else:
            # (alternatively, could use the energy differences -- here, we will use total dhdl).
            g[k] = pymbar.timeseries.statistical_inefficiency(dhdl_sum)

        indices = sta[k] + np.array(pymbar.timeseries.subsample_correlated_data(dhdl_sum, g=g[k])) # indices of uncorrelated samples
        N_uncorr = len(indices) # number of uncorrelated samples
        # Handle case where we end up with too few.
        if N_uncorr < uncorr_threshold:
            if do_dhdl:
                print("WARNING: Only %s uncorrelated samples found at lambda number %s; proceeding with analysis using correlated samples..." % (N_uncorr, k))
            indices = sta[k] + np.arange(len(dhdl_sum))
            N = len(indices)
        else:
            N = N_uncorr
        N_k[k] = N # Store the number of uncorrelated samples from state k.
        if not (u_klt is None):
            for l in range(K):
                u_kln[k,l,0:N] = u_klt[k,l,indices]
        if do_dhdl:
            print("%6s %12s %12s %12.2f" % (k, N_uncorr, N_k[k], g[k]))
            for n in range(n_components):
               dhdl[k,n,0:N] = dhdlt[k,n,indices]
    if do_dhdl:
        return (dhdl, N_k, u_kln)
    return (N_k, u_kln)

def compute_free_energy_differences(filepath: str = '.',
                            temperature: float = 298.15,
                            equiltime: float = 0.0,
                            uncorr_threshold: int = 50,
                            tol: float = 1.0e-6,
                            n_bootstraps: int = 0,
                            backend: str = 'pymbar',
                            initialize: str = 'mean-reduced-potential',
                            **kwargs,
):
    xvg_files = [str(file) for file in Path(filepath).glob("prod*.xvg")]
    if len(xvg_files) == 0:
        print("no production xvg files found in {}".format(filepath))
        return
    # Read xvg data and construct u_klt
    nsnapshots, lv, dhdlt, u_klt = readXvgData(xvg_files, temperature=temperature, equiltime=equiltime)
    # Uncorrelate xvg data
    N_k, u_kln = uncorrelate(nsnapshots, lv, dhdlt, u_klt, \
                            sta=np.zeros(u_klt.shape[0], int), fin=nsnapshots, \
                            uncorr_threshold=uncorr_threshold, do_dhdl=False)
    from pymbar.utils import kln_to_kn
    u_kn = kln_to_kn(u_kln, N_k=N_k)
    fe_diff = dict()
    if backend.lower() == 'pymbar':
        MBAR = pymbar.mbar.MBAR(u_kn, N_k, relative_tolerance = tol, n_bootstraps=n_bootstraps, initialize=initialize)
        fe_diff = MBAR.compute_free_energy_differences(uncertainty_method='svd-ew', return_theta = False)
    elif backend.lower() == 'fastmbar':
        import FastMBAR
        gpu = kwargs.get('gpu', False)
        if gpu:
            gpu_id = kwargs.get('gpu_id', 0)
            import torch
            torch.cuda.set_device(gpu_id)
        bs = True if n_bootstraps > 0 else False
        verbose = kwargs.get('verbose', False)
        FMBAR = FastMBAR.FastMBAR(energy=u_kn, num_conf=N_k, cuda=gpu, bootstrap=bs, bootstrap_num_rep=n_bootstraps, verbose=verbose)
        fe_diff['Delta_f'] = FMBAR.DeltaF
        fe_diff['dDelta_f'] = FMBAR.DeltaF_std
    else:
        raise MetaError("Unrecognized backend {} for mbar calculations".format(backend))

    return fe_diff
