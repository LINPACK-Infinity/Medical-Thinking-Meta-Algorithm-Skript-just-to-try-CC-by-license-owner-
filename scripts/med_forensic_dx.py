from __future__ import annotations
import time, json, math
import numpy as np, pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB=True
except Exception:
    HAS_XGB=False
import networkx as nx

@dataclass
class LR:
    pos: float
    neg: float

def _odds(p): return p/(1-p+1e-12)
def _from_odds(o): return o/(1+o)

class ForensicPathoEngine:
    def __init__(self, graph: nx.DiGraph, priors: dict[str,float], lr_table: dict[tuple[str,str], LR]):
        self.graph = graph; self.priors = priors; self.lr = lr_table
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Graph must be DAG.")

    def update_with_evidence(self, features: dict[str,float|int|bool], threshold_map: dict[str,float]):
        post = dict(self.priors)
        for diag in post.keys():
            odds = _odds(post[diag])
            for bef, thr in threshold_map.items():
                if (diag, bef) in self.lr:
                    v = features.get(bef, None)
                    is_pos = None
                    if isinstance(v,bool): is_pos = v
                    elif v is None: continue
                    else: is_pos = float(v) >= float(thr)
                    lr = self.lr[(diag,bef)].pos if is_pos else self.lr[(diag,bef)].neg
                    odds *= max(lr, 1e-6)
            post[diag] = _from_odds(odds)
        s = sum(post.values())+1e-12
        for k in post: post[k] = post[k]/s
        return post

    def patho_plausibility(self, selected_diag: str, features: dict[str,float|int|bool]) -> float:
        leafs = [n for n in self.graph.nodes if self.graph.out_degree(n)==0]
        paths = []
        for leaf in leafs:
            if nx.has_path(self.graph, selected_diag, leaf):
                paths.append(nx.shortest_path(self.graph, selected_diag, leaf))
        if not paths: return 0.0
        supported = 0
        for p in paths:
            bef = p[-1]; 
            if features.get(bef, None) is not None: supported += 1
        return supported/max(len(paths),1)

def build_baselines(random_state=42):
    models = {
        "logreg": LogisticRegression(max_iter=200),
        "rf": RandomForestClassifier(n_estimators=400, random_state=random_state),
    }
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=random_state, n_jobs=0
        )
    return models

def metrics_binary(y_true, y_prob, y_pred):
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"AUC":auc, "AP":ap, "Brier":brier, "Precision":p, "Recall":r, "F1":f}

def net_benefit(y_true, y_prob, thresh=0.2):
    y_pred = (y_prob>=thresh).astype(int)
    N = len(y_true)
    TP = ((y_true==1)&(y_pred==1)).sum()
    FP = ((y_true==0)&(y_pred==1)).sum()
    pt = thresh
    return (TP/N) - (FP/N)*(pt/(1-pt+1e-12))

def run_demo(df, priors, G, lr_tbl, feature_cols, label_col="y", t_std_col="t_diag_std", threshold_map=None, seed=42):
    lr_table = {(k.split("||")[0], k.split("||")[1]): LR(v["pos"], v["neg"]) for k,v in lr_tbl.items()}
    engine = ForensicPathoEngine(G, priors, lr_table)
    threshold_map = threshold_map or {}
    train, test = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=seed)

    def infer_row(row):
        feats = {c: row[c] for c in feature_cols}
        post = engine.update_with_evidence(feats, threshold_map)
        prob_true = post.get(str(row[label_col]), post.get(int(row[label_col]), 0.0)) if isinstance(row[label_col], (int,np.integer,str)) else 0.0
        pred = 1 if prob_true>=0.5 else 0
        plaus = engine.patho_plausibility(max(post, key=post.get), feats)
        return prob_true, pred, plaus

    infer = test.apply(lambda r: pd.Series(infer_row(r), index=["prob","pred","plaus"]), axis=1)
    y_true = test[label_col].values
    m_for = metrics_binary(y_true, infer["prob"].values, infer["pred"].values)
    m_for["Plaus_mean"] = float(infer["plaus"].mean())
    m_for["NB@0.2"] = net_benefit(y_true, infer["prob"].values, 0.2)

    X_train = train[feature_cols].values; y_train = train[label_col].values
    X_test  = test[feature_cols].values;  y_test  = y_true
    base = {}
    for name, model in build_baselines(seed).items():
        model.fit(X_train, y_train)
        ypb = model.predict_proba(X_test)[:,1]
        yhb = model.predict(X_test)
        mm = metrics_binary(y_test, ypb, yhb); mm["NB@0.2"] = net_benefit(y_test, ypb, 0.2)
        base[name] = mm

    return m_for, base, test, infer
