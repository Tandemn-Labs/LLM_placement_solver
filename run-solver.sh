#!/bin/bash

# ctrl+c interrupt the script, kill the background processes
trap cleanup SIGINT
trap cleanup SIGTERM
trap cleanup SIGKILL
trap cleanup SIGQUIT

function cleanup() {
    echo "** Cleaning up..."
    kill $(jobs -p)
    exit 1
}

# config_dir_list=("config/medium" "config/large")
# cost_optimization_method_list=("weighted" "enumeration")
# network_config_list=("600 400" "400 200")
# run_on_background=true

# config_dir_list=("config/medium")
# config_dir_list=("config/medium-prefill")
# config_dir_list=("config/medium-decode")
# config_dir_list=("config/large-prefill" "config/large-decode")
config_dir_list=("config/large-prefill")
# config_dir_list=("config/large-decode")

    # parser.add_argument('--sequence-length', type=int, required=True,
    #                    help='Sequence length for prefill (or KV cache length for decode)')
    # parser.add_argument('--output-length', type=int, default=0,
    #                    help='Output length for decode phase (default: 0)')
    # parser.add_argument('--min-batch-size', type=int, required=True,
    #                    help='Minimum batch size to consider')
    # parser.add_argument('--max-batch-size', type=int, required=True,
    #                    help='Maximum batch size to consider')

workload_phase_list=("prefill")
# workload_phase_list=("prefill" "decode")
# Pair input/output lengths (space-separated)
io_length_pairs=("8192 2048")
# io_length_pairs=("1024 1024" "2048 2048" "4096 2048" "8192 2048" "16384 2048")

# workload_phase_list=("prefill")
# input_token_length_list=(8192)
# output_token_length_list=(1024)

# Pair min/max batch sizes (space-separated)
batch_size_pairs=("64 64")

# cost_optimization_method_list=("weighted")
cost_optimization_method_list=("enumeration")
# Use "none" to force loading config/network_bandwidth.csv
network_config_list=("none")
run_on_background=true
cloud_provider="AWS"

timestamp=$(date +%Y%m%d_%H%M%S)
solver="solver.py"
for config_dir in "${config_dir_list[@]}"; do
    for cost_optimization_method in "${cost_optimization_method_list[@]}"; do
        for workload_phase in "${workload_phase_list[@]}"; do
            for network_config in "${network_config_list[@]}"; do
                for io_pair in "${io_length_pairs[@]}"; do
                    input_token_length=$(echo ${io_pair} | cut -d' ' -f1)
                    output_token_length=$(echo ${io_pair} | cut -d' ' -f2)
                    for batch_pair in "${batch_size_pairs[@]}"; do
                        min_batch_size=$(echo ${batch_pair} | cut -d' ' -f1)
                        max_batch_size=$(echo ${batch_pair} | cut -d' ' -f2)
                            intra_bw=$(echo ${network_config} | cut -d' ' -f1)
                            inter_bw=$(echo ${network_config} | cut -d' ' -f2)
                            output_log_dir="${config_dir}/method_${cost_optimization_method}-wrk_${workload_phase}-in${input_token_length}-out${output_token_length}-bs${min_batch_size}_${max_batch_size}-${timestamp}"
                            mkdir -p ${output_log_dir}
                            output_log_path="${output_log_dir}/output.txt"
                            echo "** Starting solver, output_log_path: ${output_log_path}"
                            start_time=$(date +%s)
                            if [ "${run_on_background}" = true ]; then
                                python3 ${solver} --config-dir ${config_dir} --output-dir ${output_log_dir} --method ${cost_optimization_method} --cloud-provider ${cloud_provider} --throughput-debug-samples 5 --sequence-length ${input_token_length} --output-length  ${output_token_length} --min-batch-size ${min_batch_size} --max-batch-size ${max_batch_size} --workload-phase ${workload_phase} &> ${output_log_path} &
                            else
                                python3 ${solver} --config-dir ${config_dir} --output-dir ${output_log_dir} --method ${cost_optimization_method} --cloud-provider ${cloud_provider} --throughput-debug-samples 5 --sequence-length ${input_token_length} --output-length  ${output_token_length} --min-batch-size ${min_batch_size} --max-batch-size ${max_batch_size} --workload-phase ${workload_phase} &> ${output_log_path}
                            fi
                            if [ "${run_on_background}" = true ]; then
                                solver_pid=$!
                                echo "** Solver PID: ${solver_pid} for config: ${config_dir} method: ${cost_optimization_method} workload: ${workload_phase} input: ${input_token_length} output: ${output_token_length} batch: ${min_batch_size} ${max_batch_size}"
                            else
                                echo "** Solver finished for config: ${config_dir} with method: ${cost_optimization_method} workload: ${workload_phase} input: ${input_token_length} output: ${output_token_length} batch: ${min_batch_size} ${max_batch_size}"
                                end_time=$(date +%s)
                                runtime=$((end_time - start_time))
                                # echo "** Solver output_log_path: ${output_log_path}, total runtime: ${runtime}"
                            fi
                    done
                done
            done
        done
    done
done