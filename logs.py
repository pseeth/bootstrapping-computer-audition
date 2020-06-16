#!/usr/bin/env python

"""
Parse the output of 'tsp' to find any running jobs.
Then tail the logs of the running jobs in
different, evenly-spaced tmux window panes.

Usage:

./logs.py | sh
"""

import subprocess

result = subprocess.check_output('tsp', shell=True, text=True)
keys = ['ID', 'State', 'Output', 'E-Level', 'Times(r/u/s)', 'Command']
loc = [[0, 0] for k in keys]

def parse_line(line):
    parsed_line = {}
    for j, key in enumerate(keys):
        end = len(line) if key == 'Command' else loc[j][1]
        parsed_line[key] = line[loc[j][0]:end].strip()
    return parsed_line

jobs = []

for i, line in enumerate(result.splitlines()):
    if i == 0:
        for j, key in enumerate(keys):
            idx = line.index(key)
            p = j - 1 if j > 0 else j
            loc[j][0] = idx
            loc[p][1] = idx 
        loc[-1][1] = -1
    else:
        jobs.append(parse_line(line))

class TmuxCommand:
    def __init__(self):
        self.command = []
    
    def split_window_and_run(self, cmd):
        self.command.append(f"split-window '{cmd}'")

    def title_panes(self):
        self.command.append(
            'set -g pane-border-format "#{pane_index} #{pane_current_command}"')
    
    def last_pane(self):
        self.command.append(f"last-pane")

    def tiled_layout(self):
        self.command.append(f"select-layout tiled")
    
    def __call__(self):
        joined_command = '\; '.join(self.command)
        command = f"tmux {joined_command}"
        print(command)

tmux_command = TmuxCommand()
tmux_command.title_panes()
for job in jobs:
    if job['State'] == 'running':
        tmux_command.split_window_and_run(f"tsp -t {job['ID']}")
        tmux_command.last_pane()
tmux_command.tiled_layout()
tmux_command()    
