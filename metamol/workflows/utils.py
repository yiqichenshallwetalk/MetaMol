import os
from pathlib import Path
from typing import Union

from dflow import Workflow, Step, Task

def download_local(wf: Workflow,
                   step: Union[Step, Task],
                   artifact_name: Union[str, list] = None,
                   local_path: str = '.',
):
    artifacts = []
    if isinstance(artifact_name, str):
        artifacts.append(artifact_name)
    elif isinstance(artifact_name, list):
        artifacts += artifact_name
    elif artifact_name == None:
        artifacts += list(step.outputs.artifacts.keys())
    Path(local_path).mkdir(exist_ok=True)
    for artifact_key in artifacts:
        artifact = step.outputs.artifacts[artifact_key]
        remote_path = os.path.join(wf.id, wf.id + '-' + step.id + '*', 'workdir' + artifact.path)

        os.system(f'cp -r {remote_path} {local_path}')
    return

def file_exists_in_directory(
    directory_path: str,
    filename: str,
) -> bool:
    """Indicates with True or False whether a filename or filename pattern
    exists in the specified directory. (But doesn't look inside subdirectories
    of that directory.)"""
    return len([file for file in Path(directory_path).glob(filename)
                if file.is_file()]) > 0

def add_path(directory_path, filename, path_list):
    if file_exists_in_directory(directory_path, filename):
        path_list.append(Path(os.path.join(directory_path, filename)))
    return

def prepare_parallel_runs(config_file: str,
                          num_runs: int = 1,
                          inp_file_prefix: str = "",
                          work_dir: str = ".",
                          distribute_to_gpu: bool = True):
    import json
    with open(os.path.join(inp_file_prefix, config_file), 'r') as f:
        FEP_configs = json.load(f)
    name = FEP_configs["name"]
    state_range = None
    for _, fep_config in FEP_configs.items():
        if "state_range" in fep_config:
            if not state_range:
                state_range = fep_config["state_range"]
            else:
                assert isinstance(state_range, list)
                assert state_range[0] == fep_config["state_range"][0] and state_range[1] == fep_config["state_range"][1]

    os.system("mkdir -p {0}".format(work_dir))
    if not state_range or num_runs <= 1:
        # no need to divide job
        out_json = open(os.path.join(work_dir, "config.json"), "w")
        json.dump(FEP_configs, out_json)
        out_json.close()
        _write_fep_run_file(os.path.join(work_dir, "fep_run.py"))

    else:
        # divide jobs
        state_st, state_end = state_range
        interval, mod = (state_end - state_st + 1) // num_runs, (state_end - state_st + 1) % num_runs
        num_states = [interval+1 if i < mod else interval for i in range(num_runs)]
        curr = state_st
        for i in range(num_runs):
            FEP_configs["name"] = name+str(i+1)
            curr_range = [curr, curr+num_states[i]-1]
            curr += num_states[i]
            for _, fep_config in FEP_configs.items():
                if "state_range" in fep_config:
                    fep_config["state_range"] = curr_range
                if distribute_to_gpu and "gmx_cmds" in fep_config:
                    extra_args = fep_config["gmx_cmds"].get("extra_args", "")
                    if "gpu" in extra_args:
                        if "-gpu_id" in extra_args:
                            extra_args_list = extra_args.split(" ")
                            for idx in range(len(extra_args_list)):
                                if extra_args_list[idx] == "-gpu_id":
                                    break
                            extra_args_list[idx+1] = str(i)
                            extra_args = " ".join(extra_args_list)
                        else:
                            extra_args += " -gpu_id " + str(i)
                        
                        fep_config["gmx_cmds"]["extra_args"] = extra_args

            curr_config_file = "config" + str(i+1) + ".json"
            out_json = open(os.path.join(work_dir, curr_config_file), "w")
            json.dump(FEP_configs, out_json, indent=4)
            out_json.close()
            _write_fep_run_file(os.path.join(work_dir, "fep_run" + str(i+1) + ".py"), 
                                config_file = curr_config_file,
                                data_dir = "data" + str(i+1))
    
    return

def _write_fep_run_file(filename: str, config_file: str = "config.json", data_dir: str = "data"):
    fepRunFile = f"""
from dflow import config
from metamol.workflows.gromacs_fep import FEP_workflow
from metamol.workflows.utils import download_local
config["mode"] = "debug"

wf, step_list = FEP_workflow(config_file="{config_file}")
wf.submit()

print(wf.query_status())

local_dir = "{data_dir}"
download_local(wf, step_list[-1], local_path=local_dir)
    """
    with open(filename, "w") as f:
        f.write(fepRunFile)

def _write_ions_mdp():
    ionsMdp = f"""
; ions.mdp - used as input into grompp to generate ions.tpr
; Parameters describing what to do, when to stop and what to save
integrator  = steep         ; Algorithm (steep = steepest descent minimization)
emtol       = 1000.0        ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep      = 0.01          ; Minimization step size
nsteps      = 200           ; Maximum number of (minimization) steps to perform

; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist         = 1         ; Frequency to update the neighbours list and long range forces
cutoff-scheme   = Verlet    ; Buffered neighbours searching
ns_type         = grid      ; Method to determine neighbours list (simple, grid)
coulombtype     = cutoff    ; Treatment of long range electrostatic interactions
rcoulomb        = 1.0       ; Short-range electrostatic cut-off
rvdw            = 1.0       ; Short-range Van der Waals cut-off
pbc             = xyz       ; Periodic Boundary Conditions in all 3 dimensions
"""
    with open('ions.mdp', 'w') as f:
        f.write(ionsMdp)