import json, numpy as np, pandas as pd, networkx as nx
from scripts.med_forensic_dx import run_demo

# Load config
priors = json.load(open("data/priors.json"))
graph = nx.readwrite.json_graph.node_link_graph(json.load(open("data/patho_graph.json")))
lr_tbl = json.load(open("data/lr_table.json"))
df = pd.read_csv("data/demo_cases.csv")
feature_cols = [c for c in df.columns if c not in ("y","t_diag_std")]
threshold_map = json.load(open("data/threshold_map.json"))

m_for, base, test, infer = run_demo(df, priors, graph, lr_tbl, feature_cols, "y", "t_diag_std", threshold_map)
print("Forensic:", m_for)
print("Baselines:", base)
