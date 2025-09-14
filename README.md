# Forensic-Patho Meta-Algorithm (TensorFlow + Bayesian Evidence)

Problem-orientierte, triage-gesteuerte medizinische Entscheidungsunterstützung nach forensisch‑pathologischem Denken.
Enthält:
- TensorFlow/Keras Meta-Architektur mit triage router, GNN-light Wissensgraph, differentiable Bayes-Update
- Policy-Head für handlungsorientierte Vorschläge (inkl. Psychopharmaka-Rezeptorprofile)
- Vergleichsbasis zu Standard-AI (LogReg/RF/XGB)
- Synthetic-Demo-Daten und Konfigs (Priors, LR, Graph)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_demo.py
```

## Medizinischer Hinweis
Forschungssoftware. Kein Medizinprodukt. Keine klinische Nutzung ohne ärztliche Prüfung.
