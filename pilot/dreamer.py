import logging
import uuid
from enum import Enum
from typing import Dict, List, Optional
from qiskit import QuantumCircuit
from math import sin, fabs, log
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger('pilot.pcs_logger')


class TaskType(Enum):
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    HYBRID = "hybrid"

class Coupling(Enum):
    TIGHT = "tight"
    MEDIUM = "medium"
    LOOSE = "loose"

class Task:
    def __init__(self, type: TaskType, resource_config: dict, args: tuple = None, kwargs: Dict = None): 
        self.type = type
        self.resource_config = resource_config
        self.task_id = kwargs.get('task_id') if kwargs and kwargs.get('task_id') else str(uuid.uuid4())
        self.args = args or ()
        self.kwargs = kwargs or {}
    
class QuantumTask(Task):
    """Represents a quantum task with requirements."""
    
    def __init__(self, circuits: List[QuantumCircuit], resource_config: Dict = None, args: tuple = None, kwargs: Dict = None):
        """
        Initialize a QuantumTask.
        
        Args:
            circuit: List of quantum circuits
            resource_config: Resource configuration dictionary
            args: Task arguments tuple
            kwargs: Task keyword arguments dictionary
        """
        super().__init__(TaskType.QUANTUM, resource_config, args, kwargs)
        self._circuits = circuits
        
    @property
    def circuits(self):
        return self._circuits

class QuantumResource:
    """Represents a quantum resource with capabilities."""
    
    def __init__(self, name: str, qubit_count: int, gateset: List[str], 
                 error_rate: float = 0.0, noise_level: float = 0.0, 
                 quantum_config: Dict = None, pilot_name: str = None):
        self.name = name
        self.qubit_count = qubit_count
        self.gateset = gateset
        self.error_rate = error_rate
        self.noise_level = noise_level
        self.quantum_config = quantum_config or {}
        self.pilot_name = pilot_name

class DreamerStrategyType(Enum):
    LEAST_ERROR_RATE = "least_error_rate"
    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"


class StrategySelector:
    """Strategy selector for resource selection."""
    def __init__(self, strategy_type: DreamerStrategyType):
        self.strategy_type = strategy_type        
        self.resources = []
    
    def select_resource(self, task: QuantumTask, resources: Dict[str, QuantumResource]) -> Optional[QuantumResource]:
        raise NotImplementedError("Strategy must implement this method")

class RoundRobinStrategy:
    """Round Robin strategy for resource selection."""
    def __init__(self):
        self.current_index = 0
    
    def select_resource(self, task: QuantumTask, resources: Dict[str, QuantumResource]) -> Optional[QuantumResource]:
        if not resources:
            return None
        resource_list = list(resources.values())
        resource = resource_list[self.current_index % len(resource_list)]
        self.current_index += 1
        return resource

class LeastErrorRateStrategy:
    """Least Error Rate strategy for resource selection."""
    def select_resource(self, task: QuantumTask, resources: Dict[str, QuantumResource]) -> Optional[QuantumResource]:
        if not resources:
            return None
        return min(resources.values(), key=lambda x: x.error_rate)

class LeastBusyStrategy:
    """Least Busy strategy for resource selection."""
    def select_resource(self, task: QuantumTask, resources: Dict[str, QuantumResource]) -> Optional[QuantumResource]:
        if not resources:
            return None
        # For now, return first available resource (can be enhanced with actual queue data)
        return list(resources.values())[0]

   


# -------- Sampling overhead factors (Qiskit add-on cutting reference) --------
_FIXED_OVERHEAD = {
    "cx": 9.0, "cz": 9.0, "cy": 9.0, "ch": 9.0, "ecr": 9.0,
    "cs": 3.0 + 2.0 * 2.0**0.5, "csdg": 3.0 + 2.0 * 2.0**0.5, "csx": 3.0 + 2.0 * 2.0**0.5,
    "iswap": 49.0, "dcx": 49.0,
}
_PARAM_GATES = {"rzz","rxx","ryy","rzx","crx","cry","crz","cphase"}

def _param_overhead(gate: str, theta: float) -> float:
    g = gate.lower()
    if g in {"rzz","rxx","ryy","rzx"}:
        return (1.0 + 2.0 * fabs(sin(theta)))**2
    if g in {"crx","cry","crz","cphase"}:
        return (1.0 + 2.0 * fabs(sin(theta/2.0)))**2
    raise ValueError(f"Unsupported parametric gate: {gate}")

def _gate_overhead(gate: str, theta: Optional[float]) -> float:
    g = gate.lower()
    if g in _FIXED_OVERHEAD:
        return _FIXED_OVERHEAD[g]
    if g in _PARAM_GATES:
        if theta is None:
            raise ValueError(f"Gate '{gate}' requires theta.")
        return _param_overhead(g, theta)
    # Unknown 2q gate → not cuttable
    raise ValueError(f"Unknown two-qubit gate for cutting: '{gate}'")

# -------- Tiny DSU --------
class DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}
        self.r = {x: 0 for x in items}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[rb] < self.r[ra]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1
        return True

@dataclass
class Edge:
    u: Any
    v: Any
    gate: str
    theta: Optional[float]
    overhead: float
    cost: float  # log(overhead)

# -------- Qiskit/dict normalization (version-safe indices) --------
def _from_qiskit_circuit(qc) -> Tuple[List[Any], List[Dict[str,Any]]]:
    n = qc.num_qubits
    nodes = [f"q{i}" for i in range(n)]
    edges: List[Dict[str,Any]] = []
    for inst, qa, _ in qc.data:
        if len(qa) != 2:
            continue
        # version-safe index
        u_idx = getattr(qa[0], "index", getattr(qa[0], "_index", None))
        v_idx = getattr(qa[1], "index", getattr(qa[1], "_index", None))
        name = getattr(inst, "name", "").lower()
        theta = None
        for p in getattr(inst, "params", []):
            if isinstance(p, (int, float)):
                theta = float(p); break
        edges.append({"u": f"q{u_idx}", "v": f"q{v_idx}", "gate": name, "theta": theta})
    return nodes, edges

def _normalize_input(circuit) -> Tuple[List[Any], List[Dict[str,Any]]]:
    if isinstance(circuit, dict) and "nodes" in circuit and "edges" in circuit:
        return circuit["nodes"], circuit["edges"]
    if hasattr(circuit, "data") and hasattr(circuit, "num_qubits"):
        return _from_qiskit_circuit(circuit)
    raise TypeError("Provide a Qiskit QuantumCircuit or dict {'nodes':..., 'edges':...}")

def _build_edges(nodes: List[Any], edges: List[Dict[str,Any]]) -> List[Edge]:
    out = []
    for e in edges:
        try:
            w = _gate_overhead(e["gate"], e.get("theta"))
        except ValueError:
            continue
        out.append(Edge(e["u"], e["v"], e["gate"], e.get("theta"), w, log(w)))
    return out

def _components(nodes: List[Any], edges: List[Edge]) -> List[List[Any]]:
    d = DSU(nodes)
    for e in edges:
        d.union(e.u, e.v)
    comp: Dict[Any, List[Any]] = {}
    for n in nodes:
        r = d.find(n)
        comp.setdefault(r, []).append(n)
    return list(comp.values())

def _mst(nodes: List[Any], edges: List[Edge]) -> List[Edge]:
    s = set(nodes)
    es = sorted([e for e in edges if e.u in s and e.v in s], key=lambda e: e.cost)
    d = DSU(nodes)
    T = []
    for e in es:
        if d.union(e.u, e.v):
            T.append(e)
        if len(T) == len(nodes) - 1:
            break
    return T

# -------- helper: filter to active qubits (degree>0) --------
def _filter_active(nodes: List[Any], built: List[Edge]) -> Tuple[List[Any], List[Edge]]:
    deg = {n: 0 for n in nodes}
    for e in built:
        deg[e.u] += 1; deg[e.v] += 1
    active_nodes = [n for n in nodes if deg[n] > 0]
    if not active_nodes:
        return [], []
    aset = set(active_nodes)
    active_edges = [e for e in built if e.u in aset and e.v in aset]
    return active_nodes, active_edges

# -------- capacity-aware assignment (greedy first-fit) --------
def _assign_fragments_by_capacity(fragments: List[List[Any]],
                                  capacities: List[int]) -> Optional[Dict[int, List[Any]]]:
    # Greedy: place largest fragments first
    frags = sorted(fragments, key=len, reverse=True)
    caps = capacities[:]  # remaining capacity
    placement: Dict[int, List[Any]] = {}
    for frag in frags:
        placed = False
        # try first-fit
        for i, cap in enumerate(caps):
            if len(frag) <= cap:
                placement.setdefault(i+1, [])
                placement[i+1] += frag
                caps[i] -= len(frag)
                placed = True
                break
        if not placed:
            return None
    # Convert to mapping QPU -> its fragment nodes (for readability keep as list)
    # (Above we may aggregate multiple frags per QPU if extra fragments > QPUs.)
    # Here, for clarity, map each QPU to the nodes it runs.
    # If you prefer 1 frag per QPU, you can keep frags instead of merging.
    return placement

# -------- main: capacity + overhead constrained planner --------
def get_cut_plan(
    circuit,
    qpus_count: int,
    qpu_capacities: Optional[List[int]] = None,   # e.g., [10,8,8,8,8] or None (uniform unlimited)
    max_overhead: Optional[float] = None,         # e.g., 1e3; None = no cap
    active_only: bool = True                      # default ON (physically meaningful)
) -> Dict[str, Any]:
    if qpus_count <= 0:
        raise ValueError("qpus_count must be >= 1")

    nodes, edge_specs = _normalize_input(circuit)
    built = _build_edges(nodes, edge_specs)

    # --- Active-only filtering (recommended) ---
    if active_only:
        nodes_active, built_active = _filter_active(nodes, built)
        if not nodes_active:
            # No 2q edges at all → nothing to cut
            return {
                "number_of_cuts": 0,
                "selected_cuts": [],
                "total_overhead": 1.0,
                "fragments": [[n] for n in nodes],
                "qpu_assignment": {1: nodes},  # everything trivial on one device
                "notes": "No active two-qubit edges; nothing to cut."
            }
        nodes, built = nodes_active, built_active

    # Initial components
    comps = _components(nodes, built)
    F0 = len(comps)

    # Build all per-component MST edges
    all_tree_edges: List[Edge] = []
    for c in comps:
        all_tree_edges += _mst(c, built)
    all_tree_edges.sort(key=lambda e: e.cost)  # cheapest first

    # We will progressively pick edges to cut, honoring max_overhead and capacity
    selected: List[Edge] = []
    disabled = set()         # edges we cut
    total_overhead = 1.0

    def _rebuild_fragments():
        d = DSU(nodes)
        for e in built:
            key = (e.u, e.v, e.gate, e.theta)
            if key in disabled or (e.v, e.u, e.gate, e.theta) in disabled:
                continue
            d.union(e.u, e.v)
        m: Dict[Any, List[Any]] = {}
        for n in nodes:
            r = d.find(n)
            m.setdefault(r, []).append(n)
        return list(m.values())

    # Helper: check if fragments can fit with some flexibility (transpilation can optimize)
    def _capacity_ok(fragments: List[List[Any]], tolerance: float = 1.2) -> bool:
        """
        Check if fragments fit within QPU capacities with tolerance for transpilation.
        tolerance=1.2 means allow 20% over capacity (transpilation might optimize it down).
        """
        if not qpu_capacities:
            return True  # no capacity constraint
        caps = [int(c * tolerance) for c in qpu_capacities]  # Apply tolerance
        assign = _assign_fragments_by_capacity(fragments, caps)
        return assign is not None

    # Helper: calculate parallelism score (more fragments = better, up to qpus_count)
    def _parallelism_score(fragments: List[List[Any]]) -> float:
        """
        Score based on parallelism: more fragments = better, but diminishing returns.
        Optimal is around qpus_count fragments for maximum parallel execution.
        """
        num_frags = len(fragments)
        if num_frags == 0:
            return 0.0
        # Score peaks at qpus_count, but more fragments can help with load balancing
        if num_frags <= qpus_count:
            return num_frags / qpus_count  # Linear up to qpus_count
        else:
            # Diminishing returns beyond qpus_count (but still useful for load balancing)
            return 1.0 + 0.5 * (num_frags - qpus_count) / qpus_count

    # Helper: calculate overall quality score (maximize parallelism, minimize overhead)
    def _quality_score(fragments: List[List[Any]], overhead: float) -> float:
        """
        Quality = parallelism_score / log(overhead)
        Higher is better: more parallelism with less overhead.
        """
        if overhead <= 0:
            return 0.0
        parallelism = _parallelism_score(fragments)
        # Use log to normalize overhead (overhead grows multiplicatively)
        overhead_penalty = log(max(overhead, 1.0))
        return parallelism / overhead_penalty if overhead_penalty > 0 else parallelism

    fragments = comps
    best_plan = None
    best_score = -1.0
    
    # Try different numbers of cuts to find optimal balance
    # Strategy: Greedily add cuts (cheapest first) and track best solution
    candidates = all_tree_edges[:]
    selected: List[Edge] = []
    disabled = set()
    total_overhead = 1.0

    # First, check if no cuts are needed (might already fit)
    if _capacity_ok(fragments):
        score = _quality_score(fragments, total_overhead)
        if score > best_score:
            best_score = score
            best_plan = {
                "fragments": fragments,
                "selected": selected[:],
                "overhead": total_overhead
            }

    # Iteratively add cuts (cheapest first) to maximize parallelism while minimizing overhead
    while candidates:
        e = candidates.pop(0)  # cheapest available cut
        # Try cutting this edge
        key = (e.u, e.v, e.gate, e.theta)
        if key in disabled or (e.v, e.u, e.gate, e.theta) in disabled:
            continue
        # Check overhead cap
        new_overhead = total_overhead * e.overhead
        if (max_overhead is not None) and (new_overhead > max_overhead):
            # skip this edge, try next one
            continue
        # Accept the cut
        disabled.add(key)
        selected.append(e)
        total_overhead = new_overhead
        # Recompute fragments
        fragments = _rebuild_fragments()
        
        # Check if this solution is feasible and better
        if _capacity_ok(fragments):
            score = _quality_score(fragments, total_overhead)
            if score > best_score:
                best_score = score
                best_plan = {
                    "fragments": fragments,
                    "selected": selected[:],
                    "overhead": total_overhead
                }
        
        # Replenish candidates with MST edges from new fragments
        more_candidates: List[Edge] = []
        for c in fragments:
            if len(c) > 1:
                more_candidates += _mst(c, built)
        # Remove those already cut
        more_candidates = [x for x in more_candidates
                           if (x.u, x.v, x.gate, x.theta) not in disabled
                           and (x.v, x.u, x.gate, x.theta) not in disabled]
        # Merge with existing and re-sort
        candidates = sorted(candidates + more_candidates, key=lambda ed: ed.cost)
        
        # Stop early if we have enough fragments and overhead is getting too high
        # (diminishing returns - more cuts won't help much)
        if len(fragments) >= qpus_count * 2 and total_overhead > 100:
            break

    # Use the best plan found (or fallback to current if no better plan)
    if best_plan is None:
        # No feasible solution found - return current state
        return {
            "number_of_cuts": len(selected),
            "selected_cuts": [{"u":e.u,"v":e.v,"gate":e.gate,"theta":e.theta,"overhead":e.overhead} for e in selected],
            "total_overhead": total_overhead,
            "fragments": fragments,
            "qpu_assignment": {},
            "notes": "No feasible solution found under constraints. Try relaxing capacity tolerance or max_overhead."
        }
    
    # Use best plan
    final_fragments = best_plan["fragments"]
    final_selected = best_plan["selected"]
    final_overhead = best_plan["overhead"]
    
    # QPU assignment: if capacities provided, use greedy assignment of fragments.
    if qpu_capacities:
        assignment = _assign_fragments_by_capacity(final_fragments, qpu_capacities) or {}
    else:
        # Map fragments to QPUs (round-robin, allowing load balancing)
        assignment = {}
        for i, frag in enumerate(final_fragments):
            qpu_id = (i % qpus_count) + 1
            assignment.setdefault(qpu_id, [])
            assignment[qpu_id] += frag

    return {
        "number_of_cuts": len(final_selected),
        "selected_cuts": [{"u":e.u,"v":e.v,"gate":e.gate,"theta":e.theta,"overhead":e.overhead} for e in final_selected],
        "total_overhead": final_overhead,
        "fragments": final_fragments,
        "qpu_assignment": assignment,
        "parallelism_score": _parallelism_score(final_fragments),
        "quality_score": best_score,
        "notes": f"Optimized for parallelism (score={best_score:.3f}) with {len(final_fragments)} fragments. "
                 f"Overhead={final_overhead:.2f}x. Capacity tolerance=20% (transpilation-aware)."
    }

# Preserve your strict interface for get_cuts()
def get_cuts(circuit, qpus_count: int) -> int:
    return get_cut_plan(circuit, qpus_count)["number_of_cuts"]



class Q_DREAMER:
    def __init__(self, quantum_resources, dreamer_type: DreamerStrategyType = DreamerStrategyType.LEAST_ERROR_RATE):
        """ QDREAMER framework provides intelligent resource selection for quantum tasks. """
        self.quantum_resources = quantum_resources
        self.strategy_type = dreamer_type
        self.strategy_selector = self.get_strategy()

    def get_strategy(self) -> StrategySelector:
        if self.strategy_type == DreamerStrategyType.LEAST_ERROR_RATE:
            return LeastErrorRateStrategy()
        elif self.strategy_type == DreamerStrategyType.ROUND_ROBIN:
            return RoundRobinStrategy()
        elif self.strategy_type == DreamerStrategyType.LEAST_BUSY:
            return LeastBusyStrategy()
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")        
    
    def get_best_resource(self, quantum_task: QuantumTask) -> Optional[QuantumResource]:
        """
        Get the best quantum resource for a given task
        
        Args:
            quantum_task: The quantum task to find best resource for
            
        Returns:
            Best quantum resource or None if no suitable resource found
        """

        best_resource = self.strategy_selector.select_resource(quantum_task, self.quantum_resources)
        print(f"Best resource selected: {best_resource.name}")
        return best_resource
    
    def get_cut_plan_for_task(
        self,
        quantum_task: QuantumTask,
        max_overhead: Optional[float] = None,
        active_only: bool = True
    ) -> Dict[str, Any]:
        """
        Get cut plan for a quantum task using available Q_DREAMER resources.
        
        Args:
            quantum_task: The quantum task with circuits to cut
            max_overhead: Maximum sampling overhead allowed (None = no limit)
            active_only: Only consider active qubits (default: True)
            
        Returns:
            Cut plan dictionary with number_of_cuts, fragments, qpu_assignment, etc.
        """
        # Get number of available QPUs from resources
        qpus_count = len(self.quantum_resources)
        
        # Extract QPU capacities from resources (qubit_count per resource)
        qpu_capacities = [resource.qubit_count for resource in self.quantum_resources.values()]
        
        # Get the first circuit from the task (or use all circuits)
        circuits = quantum_task.circuits
        if not circuits:
            raise ValueError("Quantum task has no circuits")
        
        # For now, use the first circuit (can be extended to handle multiple circuits)
        circuit = circuits[0]
        
        # Get cut plan
        return get_cut_plan(
            circuit=circuit,
            qpus_count=qpus_count,
            qpu_capacities=qpu_capacities if qpu_capacities else None,
            max_overhead=max_overhead,
            active_only=active_only
        )
    

        


 