from qiskit import QuantumCircuit
from math import pi
from pilot.dreamer import get_cuts, get_cut_plan

qc = QuantumCircuit(25)
qc.csx(0, 1); qc.cx(1, 2); qc.rzz(pi/6, 2, 3); qc.iswap(3, 4); qc.cz(0, 4)

# Basic (same as before): how many cuts to hit Q fragments, active-only by default
print(get_cuts(qc, 5))

# Capacity-aware: e.g., Q=5 QPUs with capacities [8,8,8,8,8] qubits each
plan = get_cut_plan(qc, 5, qpu_capacities=[8,8,8,8,8])
print(plan["number_of_cuts"], plan["total_overhead"], plan["qpu_assignment"])

# Overhead-constrained as well, e.g., cap total sampling overhead at 1e3
plan2 = get_cut_plan(qc, 5, qpu_capacities=[8,8,8,8,8], max_overhead=1e3)
print(plan2["notes"], plan2["number_of_cuts"], plan2["total_overhead"])
