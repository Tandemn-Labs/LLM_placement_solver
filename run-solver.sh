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

config_dir_list=("config/medium" "config/large")
# config_dir_list=("config/medium")
cost_optimization_method_list=("weighted" "enumeration")
network_config_list=("none" "600 400" "400 200")
# network_config_list=("none" "600 400")
run_on_background=true
timestamp=$(date +%Y%m%d%H%M%S)
solver="solver_constrained_with_tp-2.py"
for config_dir in "${config_dir_list[@]}"; do
    for cost_optimization_method in "${cost_optimization_method_list[@]}"; do
        for network_config in "${network_config_list[@]}"; do
            intra_bw=$(echo ${network_config} | cut -d' ' -f1)
            inter_bw=$(echo ${network_config} | cut -d' ' -f2)
            solver_log_path="${config_dir}/output-method_${cost_optimization_method}-network_${intra_bw}_${inter_bw}-${timestamp}.txt"
            echo "** Starting solver, log path: ${solver_log_path}"
            start_time=$(date +%s)
            if [ "${network_config}" != "none" ]; then
                if [ "${run_on_background}" = true ]; then
                    python3 ${solver} --config-dir ${config_dir} --method ${cost_optimization_method} --generate-network ${intra_bw} ${inter_bw} &> ${solver_log_path} &
                else
                    python3 ${solver} --config-dir ${config_dir} --method ${cost_optimization_method} --generate-network ${intra_bw} ${inter_bw} &> ${solver_log_path}
                fi
            else
                if [ "${run_on_background}" = true ]; then
                    python3 ${solver} --config-dir ${config_dir} --method ${cost_optimization_method} &> ${solver_log_path} &
                else
                    python3 ${solver} --config-dir ${config_dir} --method ${cost_optimization_method} &> ${solver_log_path}
                fi
            fi
            if [ "${run_on_background}" = true ]; then
                solver_pid=$!
                echo "** Solver PID: ${solver_pid} for config: ${config_dir} method: ${cost_optimization_method} network: ${intra_bw} ${inter_bw}"
                ## join the solver process
                # wait ${solver_pid}
                # echo "** Solver finished for config: ${config_dir} with method: ${cost_optimization_method} and network: ${intra_bw} ${inter_bw}"
                # end_time=$(date +%s)
                # runtime=$((end_time - start_time))
                # echo "** Solver output: ${solver_log_path}, total runtime: ${runtime}"
            else
                echo "** Solver finished for config: ${config_dir} with method: ${cost_optimization_method} and network: ${intra_bw} ${inter_bw}"
                end_time=$(date +%s)
                runtime=$((end_time - start_time))
                echo "** Solver output: ${solver_log_path}, total runtime: ${runtime}"
            fi
        done
    done
done