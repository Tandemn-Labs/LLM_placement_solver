# LLM_placement_solver

NOTE: You need gurobi license to run the solver.

Prerequisite:
```bash
pip install -r requirements.txt
```

## how to run the solver:

### TLDR;
run `./run-batch-sweep.sh`

change the setup(config) that you want to run in the `run-batch-sweep.sh` script. e.g., `config_dir_list=("config/medium")`
## Configurations:

#### config-dir: `config/medium`, `config/large`, `config/hal`

#### method: `weighted`, `enumeration`
- weighted: quick approximate solution
- enumeration: more accurate solution but slower

#### generate-network: `generate-network [intra_bandwidth] [inter_bandwidth]`
- intra_bandwidth: bandwidth (GB/s) within same GPU type
- inter_bandwidth: bandwidth (GB/s) between different GPU types

without specifying it, it will use the network bandwidth configuration in the config-dir/network_bandwidth.csv
