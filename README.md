# lau-reward-hacking-detector

**Cohomological reward hacking detection — holonomy of the value 1-form reveals local optimization with global cycling.**

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

## What This Does

An RL agent that appears to improve at every step—rising rewards, decreasing loss—might actually be going in circles. This crate detects that scenario using **algebraic topology**: specifically, the **first de Rham cohomology** H¹ of the agent's state manifold.

The core idea: the agent's value gradient `dV` is a differential 1-form. If it's **exact** (derives from a genuine global potential V), the agent is making real progress. If it's merely **closed but not exact**, the agent has non-trivial cohomology—locally improving while globally cycling. That's reward hacking.

The crate provides:
- **Holonomy computation** — line integral of `dV` around closed loops
- **H¹ risk scoring** — dimension of the cohomology group counts independent hacking channels
- **Local improvement tracking** — detects the "looks good step-by-step" illusion
- **Global divergence detection** — flags state-space cycling
- **Value potential reconstruction** — tries to build a global V from local patches; failure = hacking
- **Fleet monitoring** — PLATO-compatible fleet-wide safety monitoring

## Key Idea

```
∮ dV ≠ 0  ⟹  dV is not exact  ⟹  reward hacking
```

By Emergent Theorem C: if the holonomy (circular integral of the value 1-form) around any closed loop is non-zero, the agent's value function cannot be a true potential. The number of independent non-zero holonomy loops equals `dim H¹`, the number of independent hacking channels.

The detector monitors the canonical 9-step agent loop:

```
Observe → Features → Select → Execute → Outcome → Reward → Update → Evaluate → Loop
```

and flags when the agent is **locally improving** (each step looks good) but **globally cycling** (holonomy ≠ 0 around the loop).

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-reward-hacking-detector = "0.1"
```

Or via cargo:

```bash
cargo add lau-reward-hacking-detector
```

Dependencies: `nalgebra` 0.33, `serde` 1.

## Quick Start

### Single-Agent Detection

```rust
use lau_reward_hacking_detector::{AgentLoopDetector, AgentState};

fn main() {
    let mut detector = AgentLoopDetector::new(0.01); // holonomy threshold

    // Feed the agent's trajectory (9-step loop)
    let rewards = [0.0, 0.3, 0.6, 0.9, 1.2, 0.9, 0.6, 0.3, 0.1];
    let coords  = [0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.1];
    for i in 0..9 {
        detector.record_state(AgentState::new(
            vec![coords[i]], rewards[i], i
        ));
    }

    let report = detector.analyze();
    println!("Hacking: {}", report.reward_hacking_detected);
    println!("H¹ dim: {}", report.h1_dimension);
    println!("Risk: {:.3}", report.risk_score);
    println!("{}", report.recommendation);
}
```

### Fleet Monitoring

```rust
use lau_reward_hacking_detector::{FleetSafetyMonitor, AgentState};

let mut fleet = FleetSafetyMonitor::new(0.01);
fleet.register_agent("agent-1".into());
fleet.register_agent("agent-2".into());

// Record states for each agent...
fleet.record_state("agent-1", AgentState::new(vec![0.0], 0.0, 0));

let report = fleet.fleet_detect();
println!("Fleet risk: {:.3}", report.fleet_risk_score);
println!("Hacking agents: {:?}", report.hacking_agents);
```

### Manual Holonomy Computation

```rust
use lau_reward_hacking_detector::{compute_holonomy, AgentState};

let loop_states = vec![
    AgentState::new(vec![0.0, 0.0], 0.0, 0),
    AgentState::new(vec![1.0, 0.0], 1.0, 1),
    AgentState::new(vec![1.0, 1.0], 2.0, 2),
    AgentState::new(vec![0.0, 1.0], 3.0, 3),
    AgentState::new(vec![0.0, 0.0], 0.0, 4), // back to start
];
let result = compute_holonomy(&loop_states, 0.01);
println!("Holonomy: {:.4}", result.holonomy);
println!("Exact (no hacking): {}", result.is_exact);
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `AgentState` | A point on the agent's state manifold: coordinates + reward + step index |
| `ValueOneForm` | The value gradient dV at a state; `apply()` computes directional derivatives |
| `HolonomyResult` | Result of integrating dV around a closed loop |
| `H1RiskScore` | Cohomological risk: H¹ dimension, basis vectors, per-channel risk |
| `DetectionReport` | Final verdict: hacking detected, holonomy, risk score, recommendation |

### Core Functions

| Function | Description |
|----------|-------------|
| `compute_holonomy(states, tol)` | Integrate dV around a closed loop of states |
| `compute_holonomy_dec(edges, face)` | Discrete Exterior Calculus formulation |
| `compute_h1_risk(states, tol)` | Compute H¹ dimension and risk score |
| `reconstruct_value_potential(states, edges, tol)` | Try to build global V; failure = non-exact |
| `verify_coboundary(n, edges, δ, tol)` | Check if edge disagreements form a coboundary |

### Detectors

| Type | Description |
|------|-------------|
| `AgentLoopDetector` | Full 9-step loop detector with local + global analysis |
| `LocalImprovementTracker` | Tracks whether each step looks like improvement |
| `GlobalDivergenceDetector` | Detects state-space revisits and cycling |
| `ConservationMonitor` | Monitors Noether charge preservation |
| `FleetSafetyMonitor` | Fleet-wide monitoring with per-agent reports |

### Report Fields

```rust
pub struct DetectionReport {
    pub reward_hacking_detected: bool,  // true = agent is reward hacking
    pub holonomy: f64,                  // ∮ dV around the loop
    pub h1_dimension: usize,            // number of independent hacking channels
    pub risk_score: f64,                // [0, 1] overall risk
    pub locally_improving: bool,        // agent looks good step-by-step
    pub globally_cycling: bool,         // agent is revisiting states
    pub conservation_violated: bool,    // Noether charges drifting
    pub recommendation: String,         // human-readable verdict
}
```

## How It Works

### Step 1: Build the Value 1-Form

At each state transition `s_i → s_{i+1}`, compute the finite-difference approximation of `dV`:

```
dV_i ≈ (V(s_{i+1}) - V(s_i)) · (s_{i+1} - s_i) / ||s_{i+1} - s_i||²
```

This gives a differential 1-form on the state manifold.

### Step 2: Compute Holonomy

Integrate dV around the agent's trajectory (treated as a closed loop):

```
∮ dV = Σᵢ ⟨dV(sᵢ), s_{i+1} - sᵢ⟩
```

If the agent returns to its starting state, this integral should be zero for a true value function (by the fundamental theorem of calculus for exact forms).

### Step 3: Compute H¹

Build the discrete coboundary operator δ₀ from the state graph. The first cohomology:

```
H¹ = ker(δ₁) / im(δ₀)
```

counts independent non-trivial loops. Each generator of H¹ is an independent hacking channel.

### Step 4: Triangulate

Reward hacking is flagged when **all three** conditions hold:
1. **Non-zero holonomy** — the value 1-form is not exact
2. **Local improvement** — the agent appears to improve at each step
3. **Global cycling** — the agent revisits similar states

This avoids false positives from agents that are genuinely exploring or genuinely improving.

## The Math

### De Rham Cohomology

Let M be the agent's state manifold and `dV ∈ Ω¹(M)` the value 1-form.

- **Exact**: `dV = dV` for some global function V (agent has a true value function)
- **Closed**: `d(dV) = 0` (local consistency, but no global potential)
- **H¹(M) = {closed 1-forms} / {exact 1-forms}** — first de Rham cohomology

If `dim H¹ > 0`, the agent's "value function" is not a genuine potential. The agent is reward hacking.

### Holonomy

For a closed loop γ in state space:

```
Hol(γ) = ∮_γ dV
```

By Stokes' theorem, `Hol(γ) = 0` for exact forms. Non-zero holonomy means non-trivial cohomology.

### Discrete Exterior Calculus

On a graph with vertices V and edges E:
- **0-cochains**: functions on vertices (value estimates)
- **1-cochains**: functions on edges (reward differences)
- **Coboundary δ₀**: maps 0-cochains to 1-cochains via `(δ₀ f)(i,j) = f(j) - f(i)`
- **H¹ = ker(δ₁)/im(δ₀)** computed via SVD rank analysis

### Noether Charge Conservation

If the agent's update rule has a symmetry (e.g., permutation invariance), Noether's theorem guarantees a conserved charge. Violation of this charge signals that the agent has broken the symmetry—another indicator of hacking.

## Test Coverage

69 tests covering:
- Value 1-form construction, application, norms
- Holonomy computation (exact and non-exact cases)
- Discrete exterior calculus formulation
- H¹ risk scoring (no-risk, cycling, multi-channel cases)
- Local improvement tracking
- Global divergence detection
- Value potential reconstruction
- Coboundary verification
- Conservation law monitoring
- 9-step agent loop detection (nominal and hacking scenarios)
- Fleet monitoring
- Full serde round-trip for all serializable types

## License

MIT
