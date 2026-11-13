# MAS-PINN

Code for the experiments of *Efficient Modeling of Multi-Agent Systems via Structure-Aware Physics-Informed Learning*.

MAS-PINN represents every agent's trajectory with a shared neural basis and agent-specific output coefficients. Training
alternates an Adam step on the basis with a matrix-free Levenberg–Marquardt step on the coefficients (truncated CG; the
Jacobian is never assembled) and, for identification, a bounded least-squares solve for the physical parameters.

In the code and in the result files the method key of MAS-PINN is `falm`.

## Contents

| Folder | Experiment |
|---|---|
| `LV_half_observed/` | Parameter identification in a half-observed network Lotka–Volterra system; cost and scaling study |
| `Rossler_hypergraph/` | Topology and weight identification of Rössler oscillators on a hypergraph |
| `Lorenz96_forward/` | Inference (forward problem) for the Lorenz-96 system |
| `plotting/` | Figures from the saved runs |

Each experiment folder has the same layout: `util.py` (data), `network.py` (shared basis), `physics_residual.py`,
`LM_optim.py` (matrix-free LM), `run_paper.py` (runner) and a `test_*.py` script; the two identification experiments add
`inverse_QP.py` (parameter update).

## Setup

```sh
pip install -r requirements.txt
```

## Tests

```sh
cd LV_half_observed   && python test_lv.py
cd Lorenz96_forward   && python test_l96.py
cd Rossler_hypergraph && python test_rossler.py && python test_benchmark.py
```

## Running the experiments

Each trial is saved as one `.npz` file with the reconstruction, the estimates, the reference and the errors. The
relative measurement noise of the two identification experiments is 0.5%, 1% and 3%.

### Network Lotka–Volterra

```sh
cd LV_half_observed
for n in 0.005 0.01 0.03; do python run_paper.py --noise $n --device cuda; done
python ../plotting/plot_nature.py --results results --out ../figures --error-units percent

python bench_time.py && python bench_scaling.py          # cost and scaling study (GPU)
python ../plotting/plot_cost.py --scaling results/scaling.json --timing results/timing.json --out ../figures
```

### Rössler hypergraph

```sh
cd Rossler_hypergraph
python run_benchmark.py --prepare                 # datasets at the three noise levels
python run_benchmark.py --method falm             # all trials, written to results/benchmark
python ../plotting/plot_rossler.py --experiment results/benchmark
python ../plotting/plot_rossler_traj.py --experiment results/benchmark
```

### Lorenz-96

```sh
cd Lorenz96_forward
python run_paper.py                               # N = 50
python run_paper.py --problem '{"N": 100}'
python ../plotting/plot_lorenz.py                 # reference and error field, N = 100
```
