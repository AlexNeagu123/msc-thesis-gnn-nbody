"""HGNN — Hamiltonian Graph Neural Network with decomposed T + V (Bishnoi et al., 2023).

Learns H = T(v) + V(x) via separate pathways, derives dynamics via Hamilton's
equations (autodiff), and integrates with a symplectic Leapfrog step.
"""
