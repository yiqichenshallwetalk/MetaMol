import os
import sys
from subprocess import PIPE, Popen
from typing import List, Union

from metamol.utils.help_functions import cd
from metamol.exceptions import MetaError

def runCommands(
    cmds: Union[str, List[str]] = None,
    work_dir: str = None,  
    raise_error: bool = True,
    screen: bool = True, 
):
    
    if isinstance(cmds, List):
        cmds = ' '.join(cmds)

    # Go to the working directory if specified else stay in the current dir.
    if work_dir is None:
        work_dir = os.getcwd()

    with cd(work_dir):
        proc = Popen(
            cmds,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            shell=True)

        out, err = proc.communicate()
        out = out.decode(sys.stdout.encoding)
        err = err.decode(sys.stderr.encoding)
        rc = proc.poll()
        if rc != 0:
            print(err.strip())
            if raise_error:
                raise MetaError("command {0} failed.".format(cmds))
        elif (rc == 0 and screen):
            print(out.strip())
            print(err.strip())

    return rc, out, err
