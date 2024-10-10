import os
from pathlib import Path
from typing import List

from dflow import (
    Workflow,
    Task,
    upload_artifact,
    InputArtifact,
    InputParameter,
    Inputs,
    OPTemplate,
    OutputArtifact,
    OutputParameter,
    Outputs,
)
from dflow.python import OP, OPIO, Artifact, OPIOSign, PythonOPTemplate, Parameter, BigParameter

from metamol.utils.execute import runCommands
from metamol.utils.help_functions import MetaError
from metamol.workflows.utils import add_path, _write_ions_mdp

class Solvate(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "GMX": Parameter(str),
            "box": Parameter(List[float]),
            "screen": Parameter(bool, default=False),
            "add_ions": Parameter(bool, default=True),
            "ion_conc": Parameter(float, default=0.15),
            "sol_group": Parameter(int, default=15),
            "inp_gro": Artifact(Path),
            "inp_top": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "out_gro": Artifact(List[Path]),
            "out_top": Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO, ) -> OPIO:
        GMX = op_in["GMX"]
        inp_gro = op_in["inp_gro"]
        file = op_in["inp_top"]
        os.system("cp {0} .".format(str(file)))
        inp_top = str(file).split('/')[-1]
        screen = op_in["screen"]
        box = op_in["box"]

        solvate_commands = GMX + " --nobackup solvate -cp " + str(inp_gro) + " -p " + str(inp_top) + " -box " + " ".join([str(x) for x in box]) + " -o solvated.gro"
        runCommands(solvate_commands, screen=screen)
        
        # insert include water itp in top file
        with open(inp_top, 'r') as file:
            data = file.read()
        if not '#include "amber99sb-ildn.ff/tip3p.itp"' in data:
            data = data.replace('[ system ]\n', '#include "amber99sb-ildn.ff/tip3p.itp"\n\n[ system ]\n')
            with open(inp_top, 'w') as file:
                file.write(data)
        
        out_gro = [Path("solvated.gro")]
        
        if op_in["add_ions"]:
            # Add ions to the system
            _write_ions_mdp()
            runCommands(GMX + " grompp -f ions.mdp -c solvated.gro -p " + str(inp_top) + " -o ions_inp.tpr", screen=screen)
            ions_commands = "echo " + str(op_in["sol_group"]) + "| " + GMX + " genion -s ions_inp.tpr -o solvated_ions.gro -neutral -conc " + str(op_in["ion_conc"]) + " -p " + str(inp_top)
            runCommands(ions_commands, screen=screen)
            out_gro = [Path("solvated_ions.gro")]

            # insert include ions itp in top file
            if not '#include "amber99sb-ildn.ff/ions.itp"' in data:
                with open(inp_top, 'r') as file:
                    data = file.read()
                data = data.replace('[ system ]\n', '#include "amber99sb-ildn.ff/ions.itp"\n\n[ system ]\n')
                with open(inp_top, 'w') as file:
                    file.write(data)
        
        op_out = OPIO({
            "out_top": Path(inp_top),
            "out_gro": out_gro, 
        })
        
        return op_out

class PrepRunFEP(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "GMX": Parameter(str),
            "screen": Parameter(bool, default=False),
            "run_type": Parameter(str),
            #"num_states": Parameter(int),
            "state_range": Parameter(List[int]),
            "num_threads": Parameter(List[int], default=[1, 1]),
            "extra_args": Parameter(str, default=""),
            "position_restrict": Parameter(bool, default=False),
            "compute_bar": Parameter(bool, default=False),
            "inp_gro": Artifact(List[Path]),
            "inp_mdp": Artifact(Path),
            "inp_top": Artifact(Path),
            "inp_itp": Artifact(Path, optional=True),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "out_edr": Artifact(List[Path]),
            "out_xvg": Artifact(List[Path]),
            "out_log": Artifact(List[Path]),
            "out_gro": Artifact(List[Path]),
            "out_bar": Artifact(List[Path]),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO, ) -> OPIO:
        GMX = op_in["GMX"]
        screen = op_in["screen"]
        run_type = op_in["run_type"]
        #num_states = op_in["num_states"]
        state_st, state_end = op_in["state_range"]
        [ntmpi, ntomp] = op_in["num_threads"]
        extra_args = op_in["extra_args"]
        pos_restrict = op_in["position_restrict"]
        inp_gros = [Path(ii) for ii in op_in["inp_gro"]] if len(op_in["inp_gro"]) > 1 else [Path(op_in["inp_gro"][0])] * (state_end-state_st+1)
        inp_itp = op_in["inp_itp"]
        inp_top = op_in["inp_top"]
        inp_mdp = op_in["inp_mdp"]
        search_text = "state_to_set"
        Path(run_type.upper()).mkdir(exist_ok=True)
        out_gro, out_log = [], []
        out_xvg, out_edr = [], []
        for i in range(state_st, state_end+1):
            print("start running {0} simulation No {1}.".format(run_type, i))
            current_dir = os.path.join(run_type.upper(), "No_"+str(i))
            #print(Path(current_dir))
            Path(current_dir).mkdir(exist_ok=True)
            out_name = str(run_type) + "_No" + str(i)
            curr_mdp = os.path.join(current_dir, out_name+".mdp")
            replace_text = str(i)
            with open(inp_mdp, 'r') as file:
                data = file.read()
                data = data.replace(search_text, replace_text)

            with open(curr_mdp, 'w') as file:
                file.write(data)
            os.system("cp {0} {1}".format(str(inp_top), current_dir))
            if inp_itp:
                os.system("cp {0} {1}".format(str(inp_itp), current_dir))

            grompp_commands = GMX + " --nobackup grompp -f " +  out_name + ".mdp -c " + str(inp_gros[i-state_st]) + " -p " + str(inp_top.name) + " -maxwarn 99 -o " + run_type + ".tpr"
            if pos_restrict:
                grompp_commands += " -r " + str(inp_gros[i-state_st])
            runCommands(grompp_commands, work_dir = current_dir, screen=screen)

            mdrun_commands = GMX + " --nobackup mdrun -s " + run_type + ".tpr" + " -deffnm " + out_name + " -ntmpi " + str(ntmpi) + " -ntomp " + str(ntomp) + " " + extra_args
            runCommands(mdrun_commands, work_dir = current_dir, screen=screen)
            add_path(current_dir, out_name+'.log', out_log)
            add_path(current_dir, out_name+'.xvg', out_xvg)
            add_path(current_dir, out_name+'.gro', out_gro)
            add_path(current_dir, out_name+'.edr', out_edr)
            print("finished {0} simulation No {1}.".format(run_type, i))
        out_bar = []
        if op_in["compute_bar"]:
            bar_dir = os.path.join(run_type.upper(), "xvg_files")
            Path(bar_dir).mkdir(exist_ok=True)
            for i in range(state_st, state_end+1):
                current_xvg = os.path.join(run_type.upper(), "No_"+str(i), "*.xvg")
                os.system("cp {0} {1}".format(current_xvg, bar_dir))
            bar_commands = GMX + " --nobackup bar -f *.xvg -o bar_out.xvg -oi barint_out.xvg"
            runCommands(bar_commands, work_dir = bar_dir, screen=screen)
            out_bar.append(Path(os.path.join(bar_dir, "bar_out.xvg")))
            out_bar.append(Path(os.path.join(bar_dir, "barint_out.xvg")))

        return OPIO({"out_edr": out_edr,
                     "out_xvg": out_xvg,
                     "out_log": out_log,
                     "out_gro": out_gro,
                     "out_bar": out_bar,
                    })

class PrepTpr(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "GMX": Parameter(str),
            "screen": Parameter(bool, default=False),
            "run_type": Parameter(str),
            "num_states": Parameter(int),
            "inp_gro": Artifact(List[Path]),
            "inp_mdp": Artifact(Path),
            "inp_top": Artifact(Path),
            "extra_files": Artifact(Path, optional=True),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "out_mdps": Artifact(List[Path]),
            "out_tprs": Artifact(List[Path]),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO, ) -> OPIO:
        GMX = op_in["GMX"]
        screen = op_in["screen"]
        run_type = op_in["run_type"]
        num_states = op_in["num_states"]
        inp_gros = [Path(ii) for ii in op_in["inp_gro"]] if len(op_in["inp_gro"]) > 1 \
                    else [Path(op_in["inp_gro"][0])] * num_states
        extra_file = op_in["extra_files"]
        inp_top = op_in["inp_top"]
        inp_mdp = op_in["inp_mdp"]
        search_text = "state_to_set"
        Path(run_type.upper()).mkdir(exist_ok=True)
        out_mdps, out_tprs = [], []
        for i in range(num_states):
            current_dir = os.path.join(run_type.upper(), "No_"+str(i))
            #print(Path(current_dir))
            Path(current_dir).mkdir(exist_ok=True)
            curr_mdp = os.path.join(current_dir, run_type+".mdp")
            replace_text = str(i)
            with open(inp_mdp, 'r') as file:
                data = file.read()
                data = data.replace(search_text, replace_text)

            with open(curr_mdp, 'w') as file:
                file.write(data)

            os.system("cp {0} {1}".format(str(inp_top), current_dir))
            if extra_file:
                os.system("cp {0} {1}".format(str(extra_file), current_dir))

            grompp_commands = GMX + " --nobackup grompp -f " +  run_type + ".mdp -c " + str(inp_gros[i]) + " -p " + str(inp_top.name) + " -maxwarn 99 -o " + run_type + ".tpr"
            if run_type != "enmin":
                grompp_commands += " -r " + str(inp_gros[i])
            runCommands(grompp_commands, work_dir=current_dir, screen=screen)
            out_mdps.append(Path(current_dir+'/'+run_type+'.mdp'))
            out_tprs.append(Path(current_dir+'/'+run_type+'.tpr'))

        return OPIO({"out_mdps": out_mdps,
                     "out_tprs": out_tprs,
                    })

class RunFEP(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "GMX": Parameter(str),
            "screen": Parameter(bool, default=False),
            "inp_args": Parameter(str, default=""),
            "run_type": Parameter(str),
            "inp_dir": Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "out_logs": Artifact(List[Path]),
            "out_gros": Artifact(List[Path]),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO, ) -> OPIO:
        GMX = op_in["GMX"]
        screen = op_in["screen"]
        run_type = op_in["run_type"]
        inp_args = op_in["inp_args"] if op_in["inp_args"] else ""
        inp_tprs = [Path(ii) for ii in op_in["inp_dir"]]
        Path(run_type.upper()).mkdir(exist_ok=True)
        out_logs, out_gros = [], []
        for i in range(len(inp_tprs)):
            print("start running {0} simulation No {1}.".format(run_type, i))
            current_dir = os.path.join(run_type.upper(), "No_"+str(i))
            Path(current_dir).mkdir(exist_ok=True)
            out_name = str(run_type) + "_No" + str(i)


            mdrun_commands = GMX + " --nobackup mdrun -s " + str(inp_tprs[i]) + " -deffnm " + str(run_type) + "_No" + str(i) + " -ntmpi 1 -ntomp 10 -nb gpu -pin on " + inp_args
            if run_type != "enmin":
                mdrun_commands += " -pme gpu -bonded gpu -fep gpu -update gpu"

            runCommands(mdrun_commands, work_dir=current_dir, screen=screen)
            out_logs.append(Path(current_dir + '/' + out_name+'.log'))
            out_gros.append(Path(current_dir + '/' + out_name+'.gro'))

        return OPIO({"out_logs": out_logs, "out_gros": out_gros})


def FEP_workflow(config_file: str, inp_file_prefix: str = ""):
    import json
    if inp_file_prefix:
        config_file = os.path.join(inp_file_prefix, config_file)
    with open(config_file, 'r') as f:
        FEP_configs = json.load(f)
    name = FEP_configs["name"]
    GMX = FEP_configs["GMX"]
    include_solvation = "solvation" in FEP_configs
    step_list = []
    wf = Workflow(name = name.lower())
    for key, fep_config in FEP_configs.items():
        if key == "name" or key == "GMX": continue
        if key == "solvation":
            inp_gro = upload_artifact(os.path.join(inp_file_prefix, fep_config["input_data"]["inp_gro"]))
            inp_top = upload_artifact(os.path.join(inp_file_prefix, fep_config["input_data"]["inp_top"]))
            solvation = Task(name="solvate",
                             template=PythonOPTemplate(Solvate),
                             parameters={"GMX": GMX,
                                       "box": fep_config["gmx_cmds"]["box"],
                                       "screen": fep_config["gmx_cmds"]["screen"],
                                       "add_ions": fep_config["gmx_cmds"].get("add_ions", True),
                                       "ion_conc": fep_config["gmx_cmds"].get("ion_conc", 0.15),
                                       "sol_group": fep_config["gmx_cmds"].get("sol_group", 15),
                                       },
                             artifacts={"inp_gro": inp_gro, "inp_top": inp_top},)
            wf.add(solvation)
            step_list.append(solvation)
        else:
            # nvt, npt, prod runs
            inp_mdp = upload_artifact(os.path.join(inp_file_prefix, fep_config["input_data"]["inp_mdp"]))
            if "inp_itp" in fep_config["input_data"]:
                pos_resctrict = True
                inp_itp = upload_artifact(os.path.join(inp_file_prefix, fep_config["input_data"]["inp_itp"]))
            else:
                pos_resctrict = False
            if not include_solvation:
                if "inp_top" not in fep_config["input_data"]:
                    raise MetaError("Must provide topology file when not starting with a solvation step.")
                inp_top = upload_artifact(os.path.join(inp_file_prefix, fep_config["input_data"]["inp_top"]))
                if len(step_list) == 0:
                    if "inp_gro" not in fep_config["input_data"]:
                        raise MetaError("Must provide .gro file when not starting with a solvation step.")
                    inp_gro = upload_artifact(os.path.join(inp_file_prefix, fep_config["input_data"]["inp_gro"]))
                else:
                    inp_gro = step_list[-1].outputs.artifacts["out_gro"]
            else:
                inp_gro = step_list[-1].outputs.artifacts["out_gro"]
                inp_top = solvation.outputs.artifacts["out_top"]

            artifacts={"inp_gro": inp_gro,
                       "inp_top": inp_top,
                       "inp_mdp": inp_mdp}
            if pos_resctrict:
                artifacts["inp_itp"] = inp_itp
            prep_run_curr = Task(name=key+"-"+fep_config["run_type"],
                                 template=PythonOPTemplate(PrepRunFEP),
                                 parameters={"GMX": GMX,
                                            "screen": fep_config["gmx_cmds"]["screen"],
                                            "run_type": fep_config["run_type"],
                                            "state_range": fep_config["state_range"],
                                            "position_restrict": pos_resctrict,
                                            "num_threads": fep_config["gmx_cmds"]["num_threads"],
                                            "extra_args": fep_config["gmx_cmds"]["extra_args"],
                                            "compute_bar": fep_config.get("compute_bar", False)},
                                 artifacts=artifacts)
            wf.add(prep_run_curr)
            step_list.append(prep_run_curr)
    return wf, step_list
