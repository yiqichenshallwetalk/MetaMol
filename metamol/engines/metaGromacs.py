# metaGromacs object that read/write and manages Gromacs commands.
from collections import defaultdict, OrderedDict
import os
from distutils.spawn import find_executable
from subprocess import PIPE, Popen
import warnings

from metamol.utils.help_functions import cd
from metamol.utils.execute import runCommands
from metamol.exceptions import MetaError

valid_commands = ['title', 'cpp', 'integrator', 'nsteps', 'nstlist',
                'nstfout', 'nstxout', 'nstvout', 'nstxtcout', 'nstlog',
                'dt', 'constraints', 'nstenergy', 'ns_type', 'coulombtype',
                'rlist', 'rvdw', 'rcoulomb', 'tcoupl', 'tc_grps', 'tau_t',
                'ref_t', 'freezegrps', 'freezedim', 'fourier_spacing',
                'nstcalcenergy', 'cutoff-scheme']

class metaGromacs(object):

    def __init__(self):

        self.commands = defaultdict(list)
        self.numCommands = 0
        self.max_id = 0

    def close(self):
        del self

    def clear(self):
        self.commands = defaultdict(list)
        self.numCommands = 0
        self.max_id = 0

    def command(self, cmd=None):
        """Read in a single Gromacs input command from a string.

        Argument:
        ----------
        cmd: str
            a single lammps command.
        """
        if not cmd:
            return
        if not isinstance(cmd, str):
            raise TypeError("Command must be a string")

        cmd_head = cmd.split()[0]
        cmd = ' '.join(cmd.split())
        if cmd_head not in valid_commands:
            raise ValueError("Invalid command: {0}".format(cmd_head))
        if cmd in self.commands[cmd_head]:
            print("Duplicate command")
            return

        self.commands[cmd_head].append((cmd, self.max_id+1))
        self.max_id += 1
        self.numCommands += 1

    def commands_list(self, cmdlist=[]):
        """Read in multiple LAMMPS input command from a list of strings.

        Argument:
        ----------
        cmdlist: list(str)
            a list of lammps commands.
        """
        if not cmdlist:
            return
        if not isinstance(cmdlist, list) or (not isinstance(cmdlist[0], str)):
            raise TypeError("cmdlist must be a list of strings")
        for cmd in cmdlist:
            self.command(cmd)

    def file(self, filename=''):
        """Process gromacs input file.

        Argument:
        ----------
        filename: str
            path of the lammps input file to process.
        """
        if not filename:
            return
        if not os.path.exists(filename):
            raise FileNotFoundError("File {0} not found on disk".format(filename))

        cmdlist = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and line[0] != ';':
                    cmdlist.append(line)
        self.commands_list(cmdlist=cmdlist)

    # ---------------------------------------------------
    # This block of functions manages gromacs commands.

    def delete(self, cmdlist):
        """Delete a command."""
        if isinstance(cmdlist, str):
            cmdlist = [cmdlist]
        for cmd in cmdlist:
            found = False
            cmd_head = cmd.split()[0]
            cmd = ' '.join(cmd.split())
            for idx, cmd_curr in enumerate(self.commands[cmd_head]):
                if len(self.commands[cmd_head]) == 1 or cmd_curr[0] == cmd:
                    self.commands[cmd_head].pop(idx)
                    self.numCommands -= 1
                    found = True
                    break

            if not found:
                warnings.warn("Command `{0}` not found on record".format(cmd))

    def replace(self, cmd1, cmd2):
        """Replace an existing command by a new command."""
        if not (isinstance(cmd1, str) and isinstance(cmd2, str)):
            raise TypeError("Both commands must be strings")

        cmd1_head, cmd2_head = cmd1.split()[0], cmd2.split()[0]
        cmd1 = ' '.join(cmd1.split())
        cmd2 = ' '.join(cmd2.split())

        if cmd1_head != cmd2_head:
            raise MetaError("Must replace the existing command with a command of the same type")

        found = False

        for idx, (cmd_curr, cmd_id) in enumerate(self.commands[cmd1_head]):
            if len(self.commands[cmd1_head]) == 1:
                self.commands[cmd1_head][0] = (cmd2, cmd_id)
                found = True
            elif cmd_curr[0] == cmd1:
                self.commands[cmd1_head][idx] = (cmd2, cmd_id)
                found = True
                break

        if not found:
            warnings.warn("Command `{0}` not found on record".format(cmd1))

    def save(self, filepath, inorder=True, overwrite=True):
        """Save gromacs commands to a gromacs input file."""
        if not overwrite:
            if os.path.exists(filepath):
                raise FileExistsError("Target file already exists and the overwrite option is off")

        if inorder:
            command_list = [cmd_tuple for cmd_list in self.commands.values() for cmd_tuple in cmd_list]
            command_list.sort(key=lambda x: x[1])
            with open(filepath, "w") as f:
                for command in command_list:
                    f.write("{0}\n".format(command[0]))

        else:
            with open(filepath, 'w') as f:
                for cmd_head in valid_commands:
                    for command in self.commands[cmd_head]:
                        f.write("{0}\n".format(command[0]))

# ----------------------------------------------------------------------------------------------------------------------------

    def grompp(self, in_file=None, gro_file=None, top_file=None, sys=None, work_dir=None, screen=True, output=None, mpi=False, **kwargs):

        if gro_file is None and top_file is None and sys is None:
            raise MetaError("gromacs and topology files are needed for grompp command")

        # Go to the working directory if specified else stay in the current dir.
        if work_dir is None:
            work_dir = os.getcwd()

        if mpi:
            GMX = find_executable("gmx_mpi")
            if not GMX:
                raise MetaError("Gromacs mpi executable not found")
            nprocs = kwargs.get('nprocs', 1)
        else:
            GMX = find_executable("gmx") or find_executable("gmx_mpi")
            if not GMX:
                raise MetaError("Gromacs executable not found")

        with cd(work_dir):
            if sys is not None:
                if not sys.parametrized:
                    try:
                        sys.parametrize(forcefield_name='opls')
                    except:
                        sys.parametrize(backend='openmm', forcefield_files=['amber14-all.xml', 'amber14/tip3pfb.xml'])
                sys.save('sys.gro')
                sys.save('sys.top')
                gro_file, top_file = 'sys.gro', 'sys.top'

            if in_file is None:
                self.save('sys.mdp')
                in_file = 'sys.mdp'

            if mpi:
                grompp_cmd = "mpirun -np {0:d} {1} grompp -f {2} -c {3} -p {4}".format(nprocs, GMX, in_file, gro_file, top_file)
            else:
                grompp_cmd = "{0} grompp -f {1} -c {2} -p {3}".format(GMX, in_file, gro_file, top_file)

            rc, out, err = runCommands(cmds=grompp_cmd, raise_error=False, screen=screen)

            if rc != 0:
                with open("gromacs_log.txt", "w") as log_file:
                    #for line in out:
                    print(err.strip(), file=log_file)
                raise RuntimeError("GROMACS failed. See 'gromacs_log.txt'")

            if output is not None:
                with open(output, "w") as f:
                    print(out.strip(), file=f)
                    print(err.strip(), file=f)

    def mdrun(self, work_dir=None, verbose=False, confout=False, screen=True, mpi=False, extra_cmds='', **kwargs):
        """Launch a gromacs simulation in terminal."""

        # Go to the working directory if specified else stay in the current dir.
        if work_dir is None:
            work_dir = os.getcwd()

        if mpi:
            GMX = find_executable("gmx_mpi")
            if not GMX:
                raise MetaError("Gromacs mpi executable not found")
            nprocs = kwargs.get('nprocs', 1)
        else:
            GMX = find_executable("gmx")
            if not GMX:
                GMX = find_executable("gmx_mpi")
                if not GMX:
                    raise MetaError("Gromacs executable not found")

        v_cmd = '-v' if verbose else ''
        conf_cmd = '' if confout else '-noconfout'

        if mpi:
            gmx_cmd = "mpirun -np {0:d} {1} mdrun {2} {3} {4}".format(nprocs, GMX, conf_cmd, v_cmd, extra_cmds)
        else:
            gmx_cmd = "{0} mdrun {1} {2} {3}".format(GMX, conf_cmd, v_cmd, extra_cmds)

        with cd(work_dir):
            rc, out, err = runCommands(cmds=gmx_cmd, raise_error=False, screen=screen)

            # proc = Popen(
            #     gmx_cmd,
            #     stdin=PIPE,
            #     stdout=PIPE,
            #     stderr=PIPE,
            #     shell=True)

            # out = []
            # while True:
            #     line = proc.stdout.readline()
            #     if proc.poll() is not None:
            #         break
            #     if line:
            #         if screen:
            #             print(line.strip().decode())
            #         out.append(line.strip().decode())
            # rc = proc.poll()

            if rc != 0:
                with open("gromacs_log.txt", "w") as log_file:
                    #for line in err:
                    print(err.strip(), file=log_file)
                raise RuntimeError("GROMACS failed. See 'gromacs_log.txt'")

    # Execute other gmx command types        
    def run(self, cmds=None, work_dir=None, screen=True):
        if not cmds: 
            return
        rc, out, err = runCommands(cmds=cmds, work_dir=work_dir, raise_error=False, screen=screen)
        if rc != 0:
            with open("gromacs_log.txt", "w") as log_file:
                # for line in err:
                print(err.strip(), file=log_file)
            raise RuntimeError("GROMACS failed. See 'gromacs_log.txt'")

    def read_data(self, outputfile):
        import pandas as pd
        from collections import defaultdict
        #0: not reading, 1: read initial, 2: read checkpoint, 3: read average.
        read_section = 0
        read_energy = False
        self.energies = defaultdict(pd.DataFrame)
        with open(outputfile, 'r') as f:
            for line in f.readlines():
                if 'Started mdrun' in line:
                    read_section = 1
                    step = 0
                elif 'Writing checkpoint' in line:
                    read_section = 2
                    step = int(line.split()[3])
                elif 'A V E R A G E S' in line:
                    read_section = 3
                elif read_section == 3 and 'steps' in line:
                    step = int(line.split()[2])
                elif 'Energies' in line:
                    unit = line.split()[1].strip('(').strip(')')
                    if 'unit' not in self.energies: self.energies['unit'] = unit
                    read_energy = True
                    self.current_energies = pd.Series(dtype='float64')
                    self.current_energies['step'] = step
                    #self.current_energies['unit'] = unit
                    self.energy_headers = []
                elif read_energy:
                    if len(line.split()) == 0:
                        read_energy = False
                        if read_section == 3:
                            self.energies['average'] = self.energies['average'].append(self.current_energies, ignore_index=True)
                        else:
                            self.energies['checkpoint'] = self.energies['checkpoint'].append(self.current_energies, ignore_index=True)
                        read_section = 0
                    else:
                        self._read_energy(line)

    def _read_energy(self, line):
        if not self.energy_headers:
            self.energy_headers = [line[idx*15:(idx+1)*15].strip() for idx in range(len(line)//15)]
        else:
            energies = line.split()
            assert len(energies) == len(self.energy_headers)
            for idx, header in enumerate(self.energy_headers):
                self.current_energies[header] = float(energies[idx])
            self.energy_headers = []

