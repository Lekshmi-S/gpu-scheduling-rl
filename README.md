# Carbon- and Fragmentation-Aware Deep Reinforcement Learning for GPU Cluster Scheduling

A reinforcement learning-based scheduler for heterogeneous GPU clusters
that jointly considers operational carbon emissions, GPU fragmentation,
and job waiting time through spatial placement, temporal deferral, and
GPU allocation.

## Abstract

Scheduling in a GPU cluster needs to consider carbon emissions, GPU
fragmentation, and waiting time for jobs. Carbon emissions depend on
geographic location as well as the time taken to execute workloads,
while fractional GPU allocation can create resource fragmentation.
Existing schedulers have generally considered carbon-aware resource
allocation and GPU fragmentation as separate problems, rather than
jointly optimizing spatial allocation, temporal deferral, and GPU
allocation.

To explore this joint optimization problem, we investigated different
reinforcement learning techniques, including policy-gradient and
value-based methods. Based on this exploration, we propose a
**Rainbow-lite Deep Q-Network (DQN)** incorporating Double DQN, a
dueling architecture, prioritized experience replay, three-step returns,
invalid-action masking, and a node-wise shared encoder.

The scheduler models job scheduling as a sequential decision-making
task. At each decision point, it selects a feasible GPU node for a
qualified job or defers the job to a future execution slot.

In the final evaluation using a simulated 16-node heterogeneous GPU
cluster, the proposed model achieves an average operational carbon
emission of **15.944 kg**. This represents a **50.12% improvement over
Best-Fit** and a **17.41% improvement over Carbon+FGD**. The average
fragmentation ratio is **0.021**, compared with **0.034** for FGD, the
fragmentation-aware baseline. Additional ablation, carbon-prediction,
region-mapping, and Pareto analyses investigate the contribution of
individual learning components and the sensitivity of scheduling
performance.

**These results are preliminary and simulation-based.** The evaluation
uses synthesized node configurations and synthesized regional
carbon-intensity time series, with one training seed. Validation on
real-world infrastructure, actual carbon-intensity data, and multiple
training seeds is required to assess the framework's generalization
capability.

## Problem Statement

GPU cluster scheduling involves competing objectives:

-   **Operational carbon emissions:** The carbon impact of a workload
    depends on where and when it executes.
-   **GPU fragmentation:** Fractional GPU allocation can leave unusable
    or difficult-to-allocate resource fragments.
-   **Job waiting time:** Deferring jobs may reduce carbon emissions or
    improve resource availability, but can increase waiting time.

Optimizing these factors independently may lead to suboptimal
cluster-wide decisions. This project investigates their joint
optimization through reinforcement learning.

## Objectives

-   Jointly optimize spatial placement, temporal deferral, and GPU
    allocation.
-   Reduce operational carbon emissions when lower-carbon execution
    opportunities are available.
-   Reduce GPU resource fragmentation.
-   Account for the trade-off between deferral and job waiting time.
-   Evaluate the learned scheduler against baseline scheduling
    approaches.

## Proposed Approach

The scheduling problem is formulated as a sequential decision-making
task. The agent considers the current cluster and job state and chooses
among feasible scheduling actions or a deferral action.

### Rainbow-lite DQN components

The proposed value-based reinforcement learning agent combines the
following techniques:

  -----------------------------------------------------------------------
  Component                           Purpose
  ----------------------------------- -----------------------------------
  Double DQN                          Reduces overestimation bias in
                                      action-value estimates.

  Dueling architecture                Separately estimates state value
                                      and action advantages.

  Prioritized Experience Replay       Samples informative transitions
                                      more frequently.

  Three-step returns                  Propagates reward information over
                                      multiple future steps.

  Invalid-action masking              Prevents the agent from selecting
                                      infeasible scheduling actions.

  Node-wise shared encoder            Learns representations across
                                      heterogeneous GPU nodes.
  -----------------------------------------------------------------------

Policy-gradient and other value-based reinforcement learning approaches
were also explored during scheduler development.

## Evaluation Setup

-   **Environment:** Simulated heterogeneous GPU cluster
-   **Cluster size:** 16 GPU nodes
-   **Evaluation type:** Simulation-based
-   **Carbon data:** Synthesized regional carbon-intensity time series
-   **Node configurations:** Synthesized heterogeneous node setups
-   **Training seeds:** One seed in the reported final evaluation

## Results

The reported final evaluation produced the following results:

  ------------------------------------------------------------------------
  Metric                   Proposed Rainbow-lite DQN Comparison
  --------------------- ---------------------------- ---------------------
  Average operational                  **15.944 kg** 50.12% improvement
  carbon emission                                    over Best-Fit; 17.41%
                                                     over Carbon+FGD

  Average fragmentation                    **0.021** FGD reports 0.034
  ratio                                              
  ------------------------------------------------------------------------

The project also includes further analyses of:

-   **Ablation:** Examines the impact of the proposed learning
    components.
-   **Carbon prediction:** Investigates the effect of carbon-intensity
    prediction.
-   **Region mapping:** Studies the effect of geographic region
    assignment.
-   **Pareto analysis:** Examines trade-offs among scheduling
    objectives.

> **Interpretation note:** The reported improvements are based on the
> stated simulation setup. They should not be interpreted as validated
> real-world performance or as evidence of generalization across
> training seeds.

## Repository Structure

The repository is organized around the project notebooks and supporting
materials. Update this example to match the files committed to the
repository.

``` text
gpu-scheduling-rl/
├── notebooks/
│   ├── <first-notebook>.ipynb
│   └── <second-notebook>.ipynb
├── results/
│   └── <evaluation-plots-and-results>
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/Lekshmi-S/gpu-scheduling-rl.git
cd gpu-scheduling-rl
```

### 2. Create and activate a virtual environment

``` bash
python -m venv .venv
```

On Windows:

``` bash
.venv\Scripts\activate
```

On macOS or Linux:

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the notebooks

Launch Jupyter:

``` bash
jupyter notebook
```

Open the notebooks in the `notebooks/` directory and execute the cells
in their intended order. If the notebooks depend on generated data,
configuration files, or checkpoints, ensure those files are available at
the paths expected by the notebooks.

## Reproducibility and Limitations

The current results are preliminary and are subject to the following
limitations:

1.  The cluster node configurations are synthesized.
2.  Regional carbon-intensity time series are synthesized rather than
    sourced from live or historical real-world measurements.
3.  The reported final evaluation uses one training seed, so variability
    across independent runs has not yet been established.
4.  Real-world infrastructure validation has not been performed.

Further work should evaluate the scheduler using real GPU cluster traces
and infrastructure, actual regional carbon-intensity data, and multiple
independent training seeds. Reporting variability and statistical
comparisons across runs would help establish the robustness and
generalization of the approach.

## Technologies

The project uses Python and deep reinforcement learning techniques. Add
the exact libraries, frameworks, and versions used in the implementation
to `requirements.txt`.

## Citation

If you use or discuss this work, please cite the associated paper or
technical report once its bibliographic details are available.

## License

A license has not yet been specified. Add a `LICENSE` file and update
this section before distributing or reusing the code. If the repository
is intended to remain proprietary, do not add an open-source license.
