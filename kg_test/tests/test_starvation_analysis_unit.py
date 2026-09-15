from graft.starvation_analysis import recovery_curve, paired_delta

ROWS = [
    {"concept":"A","level":1,"draw":0,"condition":"hub","dino":0.5},
    {"concept":"A","level":1,"draw":0,"condition":"rawnn","dino":0.4},
    {"concept":"B","level":1,"draw":0,"condition":"hub","dino":0.6},
    {"concept":"B","level":1,"draw":0,"condition":"rawnn","dino":0.6},
    {"concept":"A","level":5,"draw":0,"condition":"hub","dino":0.7},
]

def test_recovery_curve_groups_by_level():
    curve = recovery_curve(ROWS, "dino")
    assert curve["hub"][1] == (0.5 + 0.6) / 2
    assert curve["hub"][5] == 0.7

def test_paired_delta_hub_vs_rawnn():
    d = paired_delta(ROWS, "hub", "rawnn", level=1, metric="dino")
    assert d["n"] == 2 and abs(d["mean_delta"] - 0.05) < 1e-9   # (+0.1, 0.0)/2
    assert d["win_rate"] == 0.5                                  # A wins, B ties
