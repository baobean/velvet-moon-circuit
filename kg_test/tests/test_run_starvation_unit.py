from graft.run_starvation import plan_commands
from graft.starvation import CONDITIONS, iter_cells

def test_plan_one_command_per_cell_and_condition():
    cmds = plan_commands("g.json", "outputs", "data/treevill/rawdata2", "out", "c.yaml")
    assert len(cmds) == len(list(iter_cells())) * len(CONDITIONS)
    sample = cmds[0]
    assert "--condition" in sample and "--graph" in sample and "g.json" in sample
