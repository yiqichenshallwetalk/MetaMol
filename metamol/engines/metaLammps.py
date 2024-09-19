# metaLammps object that read/write and manages lammps commands.
from collections import defaultdict, OrderedDict
import os
from distutils.spawn import find_executable
from subprocess import PIPE, Popen
import tempfile
from typing import Iterable
import warnings
import pandas as pd

from metamol.utils.help_functions import cd
from metamol.utils.execute import runCommands
from metamol.exceptions import MetaError

valid_commands = ['clear', 'units', 'dimension', 'boundary', 'atom_style', 'plugin', 'pair_style',
    'pair_modify', 'special_bonds', 'bond_style', 'angle_style', 'dihedral_style',
    'improper_style', 'read_data', 'read', 'read_restart', 'include', 'lattice', 'region', 
    'create_box', 'change_box', 'create_atom', 'atom_modify', 'mass', 'delete_atoms', 
    'delete_bonds', 'group', 'velocity', 'kspace_style', 'kspace_modify', 'neighbor', 
    'neigh_modify', 'compute', 'fix', 'fix_modify', 'unfix', 'thermo', 'thermo_style', 
    'thermo_modify', 'min_style', 'minimize', 'dump', 'undump', 'timestep', 'reset_timestep',
    'run_style', 'run', 'write_data', 'write_restart']

# thermo_style_map = {'Step': 'step', 'Temp': 'temp', 'Press': 'press', 'Volume': 'vol', 'Density': 'density',
#                 'PotEng': 'pe', 'KinEng': 'ke', 'E_vdwl': 'evdwl', 'E_coul': 'ecoul',
#                 'E_pair': 'epair', 'E_bond': 'ebond', 'E_angle': 'eangle', 'E_dihed': 'edihed',
#                 'E_impro': 'eimp', 'E_mol': 'emol', 'E_long': 'elong', 'TotEng': 'etotal', 'Atoms': 'atoms'}

class metaLammps(object):

    def __init__(self, cmdargs=None):

        self.cmdargs = ''
        if isinstance(cmdargs, str):
            self.cmdargs = cmdargs
        self.commands = defaultdict(list)
        self.numCommands = 0
        self.max_id = 0
        self.thermo = OrderedDict()

    def close(self):
        """close the metaLAMMPS instance"""
        self.clear()
        del self

    def clear(self):
        """Clear all commands"""
        self.commands = defaultdict(list)
        self.numCommands = 0

    def command(self, cmd=None):
        """Process a single LAMMPS input command from a string.

        Arguments:
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
        # if cmd_head not in valid_commands:
        #     raise ValueError("Invalid command: {0}".format(cmd_head))
        if cmd in self.commands[cmd_head]:
            print("Duplicate command")
            return

        self.commands[cmd_head].append((cmd, self.max_id+1))
        self.max_id += 1
        self.numCommands += 1

    def commands_list(self, cmdlist=[]):
        """Process multiple LAMMPS input command from a list of strings.

        Arguments:
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
        """Process lammps input file.

        Arguments:
        ---------- 
        filename: str
            path of the lammps input file to process.
        """    
        if not filename:
            return
        if not os.path.exists(filename):
            raise FileNotFoundError("File {0} not found on disk".format(filename))

        cmdlist = []
        concat = False
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line: continue
                if concat:
                    line = prev_line + line
                if line[-1] == '&':
                    prev_line = line[:-1]
                    concat = True
                else:
                    concat = False
                if line[0] != '#' and not concat:
                    cmdlist.append(line)
        self.commands_list(cmdlist=cmdlist)

    def get_commands(self, style=[]):
        """Return commands with given id or style"""
        output = []
        if isinstance(style, str):
            style = [style]

        for s in style:
            if s in self.commands:
                output += self.commands[s]
        return output
    # ---------------------------------------------------
    # This block of functions manage lammps commands.
    # Please note, these functions can only be used to alter the
    # commands record which can be written to a file. They cannot
    # undo the commands that were already executed by lammps.

    def delete(self, cmdlist):

        if isinstance(cmdlist, str):
            cmdlist = [cmdlist]
        if not isinstance(cmdlist, Iterable):
            raise TypeError("Lammps command object must be a string or a list")

        for cmd in cmdlist:
            found = False
            cmd_split = cmd.split()
            cmd_head = cmd_split[0]

            if len(cmd_split) == 1:
                # if commnad input has only head, delete all commands with the sprcified head.
                if self.commands[cmd_head]:
                    found = True
                    self.numCommands -= len(self.commands[cmd_head])
                    del self.commands[cmd_head]
            else:
                cmd = ' '.join(cmd_split)
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
        """Save lammps commands to a lammps input file."""
        if not overwrite:
            if os.path.exists(filepath):
                raise FileExistsError("Target file already exists and the overwrite option is off")
        
        if inorder:
            command_list = [cmd_tuple for cmd_list in self.commands.values() for cmd_tuple in cmd_list]
            command_list.sort(key=lambda x: x[1])
            with open(filepath, "w") as f:
                for command in command_list:
                    f.write("{0}\n\n".format(command[0]))                   
        else:
            with open(filepath, 'w') as f:
                for cmd_head in valid_commands:
                    for command in self.commands[cmd_head]:
                        f.write("{0}\n\n".format(command[0]))

# ---------------------------------------------------

    def launch(self, lmp_cmd=None, work_dir=None, input=None, screen=True, output=None, inorder=True, mpi=False, gpu=False, gpu_backend='kokkos', read_thermo=True, **kwargs):
        """Launch a lammps simulation in terminal.
            If the simulation is run in parallel using MPI,
            then you cannot create new metaLammps instance after
            this step since the MPI environment needs to be finalized."""
        
        if self.numCommands == 0:
            raise MetaError("No commands are passed to metaLammps. Unable to launch.")

        # Go to the working directory if specified else stay in the current dir.
        if work_dir is None:
            work_dir = os.getcwd()

        if 'cmdargs' in kwargs:
            self.cmdargs += ' ' + kwargs['cmdargs']

        if mpi:
            LMP = find_executable("lmp_mpi") or find_executable("lmp")
            if not LMP:
                    raise MetaError("Lammps mpi executable not found")
            nprocs = kwargs.get('nprocs', 1)
        else:
            LMP = find_executable("lmp_serial") or find_executable("lmp")
            if not LMP:
                raise MetaError("Lammps executable not found")
        
        with cd(work_dir):

            if input is None:
                temp_file = tempfile.NamedTemporaryFile(suffix=".in", delete=False)
                self.save(temp_file.name, inorder=inorder, overwrite=True)
            else:
                overwrite = kwargs.get('overwrite', True)
                if os.path.exists(input) and not overwrite:
                    raise FileExistsError("Target input file path already exists and the overwrite option is off")
                self.save(input, inorder=inorder, overwrite=overwrite)

            if not lmp_cmd:
                # lmp_cmd not specified, construct it here
                gpu_cmd = ""
                if gpu:
                    if gpu_backend == 'kokkos':
                        num_gpus = kwargs.get("num_gpus", 1)
                        gpu_cmd = "-k on g "+str(num_gpus)+" -sf kk" + " -pk kokkos newton on neigh half"
                    elif gpu_backend == 'gpu':
                        gpu_cmd = "-sf gpu"
                    else: 
                        raise MetaError("Unspoorted GPU backend")

                lmp_in = input if input else temp_file.name
                if mpi:
                    lmp_cmd = "mpirun -np {0:d} --oversubscribe {1} {2} -in {3} {4}".format(nprocs, LMP, gpu_cmd, lmp_in, self.cmdargs)
                else:
                    lmp_cmd = "{0} {1} -in {2} {3}".format(LMP, gpu_cmd, lmp_in, self.cmdargs)

            rc, out, err = runCommands(cmds=lmp_cmd, raise_error=False, screen=screen)

            if input is None:
                os.remove(temp_file.name)

            if rc != 0:
                with open("lammps_log.txt", "w") as log_file:
                    #for line in out:
                    print(err.strip(), file=log_file)
                raise RuntimeError("LAMMPS failed. See 'lammps_log.txt'")
            
            if output is not None:
                with open(output, "w") as f:
                    #for line in out:
                    print(out.strip(), file=f)
                    print(err.strip(), file=f)

            if read_thermo:
                log_exists = os.path.exists("log.lammps")
                if not (log_exists or output):
                    warnings.warn("No outputs are written to file. Couldnot read thermo data.")
                    return
                elif log_exists: 
                    read_file = "log.lammps"
                else: 
                    read_file = output

                self.read_thermo_data(read_file)

    def read_thermo_data(self, read_file, write_file=None):
        current_run = 1
        columns = []
        in_run = False

        with open(read_file, "r") as f:
            for line in f.readlines():
                if line.startswith("Per MPI rank memory allocation"):
                    in_run = True
                elif in_run and len(columns)==0:
                    columns = line.split()
                    for idx, col in enumerate(columns):
                        columns[idx] = col
                        # if col in thermo_style_map:
                        #     columns[idx] = thermo_style_map[col]
                        # else:
                        #     columns[idx] = col
                    #columns = [thermo_style_map[col] for col in columns if col in thermo_style_map else col.lower()]
                    self.thermo["r"+str(current_run)] = pd.DataFrame(columns=columns)
                
                elif line.startswith("Loop time of "):
                    in_run = False
                    columns = []
                    current_run += 1

                elif in_run and len(columns) > 0:
                    items = line.split()
                    if len(items) == len(columns):
                        try:
                            vals = [float(x) for x in items]
                            self.thermo["r"+str(current_run)].loc[self.thermo["r"+str(current_run)].size] = vals
                        except ValueError:
                            pass
        for df in self.thermo.values():
            df.reset_index(inplace=True, drop=True)

        if write_file:
            #write_file = "Thermo_Data.csv"
            for run, df in self.thermo.items():
                if run=="r1":
                    df.to_csv(write_file, mode="w", header=True, index=False)
                else:
                    df.to_csv(write_file, mode="a", header=True, index=False)

    def get_thermo(self, props="all", runs="all", steps="all"):
        if props == None: return
        if not self.thermo:
            raise MetaError("No thermo stats available yet.")
        
        if runs=="all":
            runs = list(self.thermo.keys())
        elif isinstance(runs, str):
            if not runs.startswith("r"): 
                raise MetaError("Invalid name of run")
            runs = [runs]
        elif not isinstance(runs, list):
            raise TypeError("Invalid datatype of runs")
        
        if steps!="all" and isinstance(steps, str):
            if not steps.isdigit():
                raise TypeError("Step must be an integer")
            steps = [int(steps)]
        elif isinstance(steps, int):
            steps = [steps]
        elif steps!="all" and (not isinstance(steps, list)):
            raise TypeError("Invalid datatype of step")

        if props != "all" and isinstance(props, str):
            props = [props]
        elif props != "all" and (not isinstance(props, list)):
            raise TypeError("Invalid datatype of property")

        #props = [p.lower() for p in props]
        if 'Step' not in props: props = ['Step'] + props

        keys = list(self.thermo.keys())
        cols = props if props!="all" else list(self.thermo[keys[0]].column)
        output = pd.DataFrame(columns=cols)
        for run in keys:
            if run not in runs:
                continue
            df = self.thermo[run]
            for idx, row in df.iterrows():
                if steps!="all" and int(row['Step']) not in steps:
                    continue
                if props=="all":
                    vals = row
                else:
                    vals = [row[p] for p in props if p in row.index]
                output.loc[output.size] = vals
        output.reset_index(inplace=True, drop=True)
        return output

    def __getattr__(self, __name: str):
        return [cmd[0] for cmd in self.commands[__name]]

    