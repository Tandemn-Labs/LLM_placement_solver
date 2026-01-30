#!/usr/bin/env python3
"""
Collect eval-mode results into a single CSV.

Searches for placement_metrics.json / placement_metrics.csv under given roots.
Augments fields by parsing directory naming convention from run-eval-sweep.sh
and optional output.txt logs.
"""

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional


DIR_PATTERN = re.compile(
    r"eval_family-(?P<family>[^-]+)-pp(?P<pp>\d+)-tp(?P<tp>\d+)-"
    r"in(?P<input>\d+)-out(?P<output>\d+)-bs(?P<min>\d+)_(?P<max>\d+)"
)


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_csv_first_row(path: str) -> Optional[Dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return next(reader, None)
    except Exception:
        return None


def parse_output_txt(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Workload phase:" in line:
                data["workload_phase"] = line.strip().split("Workload phase:", 1)[1].strip()
            if "ERROR - Solver failed:" in line:
                data["error"] = line.strip().split("ERROR - Solver failed:", 1)[1].strip()
    return data


def parse_dir_metadata(dir_path: str) -> Dict[str, Optional[str]]:
    data: Dict[str, Optional[str]] = {
        "instance_family": None,
        "pp_stages": None,
        "tp_degree": None,
        "input_length": None,
        "output_length": None,
        "min_batch_size": None,
        "max_batch_size": None,
    }
    match = DIR_PATTERN.search(dir_path)
    if not match:
        return data
    data.update({
        "instance_family": match.group("family"),
        "pp_stages": match.group("pp"),
        "tp_degree": match.group("tp"),
        "input_length": match.group("input"),
        "output_length": match.group("output"),
        "min_batch_size": match.group("min"),
        "max_batch_size": match.group("max"),
    })
    return data


def summarize_from_json(blob: Dict[str, Any]) -> Dict[str, Any]:
    config = blob.get("config", {})
    solution = blob.get("solution", {})
    placement = blob.get("placement", {})
    assignments = solution.get("gpu_assignments", [])
    connections = solution.get("network_connections", [])

    gpu_types = sorted({a.get("gpu_type") for a in assignments if a.get("gpu_type")})
    instance_families = sorted({gt.split("#", 1)[0] for gt in gpu_types})
    tp_degrees = sorted({a.get("tp_degree") for a in assignments if a.get("tp_degree")})

    num_gpus = sum(len(a.get("gpu_ids", [])) for a in assignments)
    num_instances = len(assignments)

    placement_stages = placement.get("stages", [])
    if not assignments and placement_stages:
        num_instances = len(placement_stages)
        num_gpus = sum(len(stage.get("gpu_ids", [])) for stage in placement_stages)
        gpu_types = sorted({stage.get("gpu_type") for stage in placement_stages if stage.get("gpu_type")})
        instance_families = sorted({gt.split("#", 1)[0] for gt in gpu_types})
        tp_degrees = sorted({stage.get("tp_degree") for stage in placement_stages if stage.get("tp_degree")})

    throughput = solution.get("throughput_tokens_per_sec")
    raw_throughput = solution.get("raw_throughput_tokens_per_sec")
    pipeline_eff = solution.get("pipeline_efficiency")
    cost_per_hour = solution.get("cost_per_hour")
    cost_per_token = solution.get("cost_per_token")

    if raw_throughput and pipeline_eff:
        real_world_eff = (
            throughput / (raw_throughput * pipeline_eff) if throughput and raw_throughput > 0 else None
        )
    else:
        real_world_eff = None

    min_net_tp = None
    if connections:
        net_tps = [c.get("throughput") for c in connections if c.get("throughput") is not None]
        if net_tps:
            min_net_tp = min(net_tps)

    status = solution.get("status") or ("SUCCESS" if solution.get("throughput_tokens_per_sec") else None)
    batch_size = solution.get("batch_size")
    if batch_size is None:
        batch_size = placement.get("batch_size") or config.get("min_batch_size")

    status = solution.get("status") or ("SUCCESS" if throughput else None)
    return {
        "model_name": config.get("model_name"),
        "num_decoder_layers": config.get("num_decoder_layers"),
        "sequence_length": config.get("sequence_length"),
        "output_length": config.get("output_length"),
        "batch_size": batch_size,
        "workload_phase": placement.get("workload_phase"),
        "throughput_tokens_per_sec": throughput,
        "raw_throughput_tokens_per_sec": raw_throughput,
        "pipeline_efficiency": pipeline_eff,
        "real_world_efficiency": real_world_eff,
        "cost_per_hour": cost_per_hour,
        "cost_per_token": cost_per_token,
        "dollar_per_million_token": cost_per_token * 1_000_000 if cost_per_token is not None else None,
        "total_runtime_hours": solution.get("total_runtime_hours"),
        "num_pipeline_stages": solution.get("num_pipeline_stages"),
        "num_instances": num_instances,
        "num_gpus": num_gpus,
        "gpu_types": ",".join(gpu_types),
        "instance_families": ",".join(instance_families),
        "tp_degrees": ",".join(str(v) for v in tp_degrees if v is not None),
        "min_network_throughput": min_net_tp,
        "throughput_per_gpu": (throughput / num_gpus) if throughput and num_gpus else None,
        "throughput_per_dollar": (throughput / cost_per_hour) if throughput and cost_per_hour else None,
        "status": status,
        "error": solution.get("error"),
    }


def summarize_from_csv(row: Dict[str, str]) -> Dict[str, Any]:
    def to_float(value: Optional[str]) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except Exception:
            return None

    throughput = to_float(row.get("total_tokens_per_sec"))
    cost_per_hour = to_float(row.get("cost_per_hour"))

    return {
        "model_name": row.get("model_name"),
        "num_decoder_layers": to_float(row.get("total_layers")),
        "sequence_length": to_float(row.get("max_input_length")),
        "output_length": to_float(row.get("max_output_length")),
        "batch_size": to_float(row.get("batch_size")),
        "throughput_tokens_per_sec": throughput,
        "raw_throughput_tokens_per_sec": None,
        "pipeline_efficiency": None,
        "real_world_efficiency": None,
        "cost_per_hour": cost_per_hour,
        "cost_per_token": None,
        "dollar_per_million_token": to_float(row.get("dollar_per_million_token")),
        "total_runtime_hours": to_float(row.get("total_runtime_hours")),
        "num_pipeline_stages": to_float(row.get("pipeline_stages") or row.get("pp")),
        "num_instances": None,
        "num_gpus": to_float(row.get("num_gpus")),
        "gpu_types": row.get("device_type") or "",
        "instance_families": "",
        "tp_degrees": row.get("tp_per_stage") or "",
        "min_network_throughput": None,
        "throughput_per_gpu": (throughput / to_float(row.get("num_gpus"))) if throughput and row.get("num_gpus") else None,
        "throughput_per_dollar": (throughput / cost_per_hour) if throughput and cost_per_hour else None,
        "status": row.get("status") or "SUCCESS",
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect eval results into one CSV.")
    parser.add_argument("--input-dir", required=True, help="Parent directory to search")
    args = parser.parse_args()

    records: List[Dict[str, Any]] = []
    seen_dirs = set()
    specs_by_instance = {}
    for probe in (
        os.path.join(args.input_dir, "cloud_instances_specs.csv"),
        os.path.join(args.input_dir, "config", "cloud_instances_specs.csv"),
        os.path.join(os.path.dirname(args.input_dir), "cloud_instances_specs.csv"),
        os.path.join(os.path.dirname(args.input_dir), "config", "cloud_instances_specs.csv"),
    ):
        if os.path.exists(probe):
            with open(probe, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("Instance Name") or row.get("instance_name") or "").strip()
                    gpu_model = (row.get("GPU Model") or row.get("gpu_model") or "").strip()
                    if name and gpu_model:
                        specs_by_instance[name] = gpu_model
            break

    for dirpath, _, filenames in os.walk(args.input_dir):
        if ("placement_metrics.json" in filenames or "placement_metrics.csv" in filenames
                or "output.txt" in filenames):
            if dirpath in seen_dirs:
                continue
            seen_dirs.add(dirpath)

            json_path = os.path.join(dirpath, "placement_metrics.json")
            csv_path = os.path.join(dirpath, "placement_metrics.csv")
            output_txt = os.path.join(dirpath, "output.txt")

            payload = read_json(json_path) if os.path.exists(json_path) else None
            if payload:
                summary = summarize_from_json(payload)
            else:
                row = read_csv_first_row(csv_path) if os.path.exists(csv_path) else None
                if row:
                    summary = summarize_from_csv(row)
                else:
                    log_meta = parse_output_txt(output_txt)
                    summary = {
                        "model_name": None,
                        "num_decoder_layers": None,
                        "sequence_length": None,
                        "output_length": None,
                        "batch_size": None,
                        "workload_phase": log_meta.get("workload_phase"),
                        "throughput_tokens_per_sec": None,
                        "raw_throughput_tokens_per_sec": None,
                        "pipeline_efficiency": None,
                        "real_world_efficiency": None,
                        "cost_per_hour": None,
                        "cost_per_token": None,
                        "dollar_per_million_token": None,
                        "total_runtime_hours": None,
                        "num_pipeline_stages": None,
                        "num_instances": None,
                        "num_gpus": None,
                        "gpu_types": "",
                        "instance_families": "",
                        "tp_degrees": "",
                        "min_network_throughput": None,
                        "throughput_per_gpu": None,
                        "throughput_per_dollar": None,
                        "status": "ERROR",
                        "error": log_meta.get("error"),
                    }

            summary["result_dir"] = os.path.relpath(dirpath, args.input_dir)

            dir_meta = parse_dir_metadata(dirpath)
            summary.update({
                "instance_family": dir_meta.get("instance_family"),
                "pp_stages": dir_meta.get("pp_stages"),
                "tp_degree": dir_meta.get("tp_degree"),
                "input_length": dir_meta.get("input_length"),
                "output_length": dir_meta.get("output_length"),
                "min_batch_size": dir_meta.get("min_batch_size"),
                "max_batch_size": dir_meta.get("max_batch_size"),
            })

            log_meta = parse_output_txt(output_txt)
            summary.update(log_meta)

            instance_family = summary.get("instance_family")
            device_type = None
            if instance_family:
                device_type = specs_by_instance.get(instance_family)
            summary["device_type"] = device_type

            records.append(summary)

    if not records:
        print("No placement_metrics.json/csv found under provided roots.")
        return 1

    for rec in records:
        rec.pop("instance_families", None)
    fieldnames = sorted({key for rec in records for key in rec.keys()})
    output_path = os.path.join(args.input_dir, "eval_results.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

