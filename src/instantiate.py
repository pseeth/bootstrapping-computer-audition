import itertools
import os
import gin
import logging

@gin.configurable
def sweep(parameters):
    keys = sorted(list(parameters.keys()))
    values = [parameters[k] for k in keys]
    cartesian_product = itertools.product(*values)
    sweep_as_dict = {}

    for setting in cartesian_product:
        for i, v in enumerate(setting):
            gin.bind_parameter(keys[i], v)
            sweep_as_dict[keys[i]] = v
        sweep_as_str = [
            f"{k.split('.')[-1]}:{v}"
            for k, v in sweep_as_dict.items()
        ]
        sweep_as_str = '_'.join(sweep_as_str)
        yield sweep_as_str

def _format_task_spooler_script(output_paths):
    commands = ['#!/bin/sh']
    for o in output_paths:
        _cmd = f"tsp ./allocate.py 1 python main.py all -c {o}"
        commands.append(_cmd)
        _cmd = f"echo Queuing {commands[-1]} and sleeping while it starts..."
        commands.append(_cmd)
        _cmd = f"sleep 30"
        commands.append(_cmd)
        
    commands = '\n'.join(commands)

    command_path = os.path.abspath(
        os.path.relpath(
            os.path.join(
                output_paths[0], 
                os.path.relpath('../..')
            )
        )
    ) 

    with open(os.path.join(command_path, 'run.sh'), 'w') as f:
        f.write(commands)

def instantiate(folder):
    os.makedirs(folder, exist_ok=True)

    def get_run_number(path):
        return len([
            x for x in os.listdir(path) 
        ])

    def write_gin_config(output_folder, swp=''):
        if swp is not '':
            swp = swp.lower()
            for c in " \{\}[]'":
                swp = swp.replace(c, '')
            for c in ",:":
                swp = swp.replace(c, '_')
            output_folder += f':{swp}'
        os.makedirs(output_folder, exist_ok=True)
        gin.bind_parameter(
            'output_folder._output_folder', 
            os.path.abspath(output_folder)
        )
        output_path = os.path.join(output_folder, 'config.gin')
        with open(output_path, 'w') as f:
            logging.info(f'{swp} -> {output_path}')
            f.write(gin.config_str())
        return output_path

    run_number = get_run_number(folder) 
    output_paths = []

    try:
        for i, swp in enumerate(sweep()):
            output_folder = os.path.join(
                folder, f'run{run_number + i}')
            output_path = write_gin_config(output_folder, swp)
            output_paths.append(output_path)
    except:
        output_folder = os.path.join(
            folder, f'run{run_number}')
        output_path = write_gin_config(output_folder)
        output_paths.append(output_path)

    _format_task_spooler_script(output_paths)
