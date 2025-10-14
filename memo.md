# Memo

10 layers

GPU type A: 4
GPU type B: 4

layer id: 0,1,2,3,4,5,6,7,9

TP=1
different col means different PP layers
A A A A (gpu type)
1 2 3 4 (# layers)
2 2 2 4
3 3 3 2
2 3 3 3

higher batch size == higher throughput not true
$/token, throughput/$
batch size = 10

A+B, TP=1
A B <— B is more performant than A and the memory size same, 
3 7 * 

A B B, same batch size -> does not make sense. it makes sense only if we increases the batch size.
2 3 3

TP1,2, or 4
Ax4
Bx4

option 1: Ax4 TP=4
option 2: Bx4 TP=4 (easy, chose better $/throughput gpu -> A)

option 1: Ax4 TP=4

option 3: PP=2: Ax2 TP=2, Bx2 TP=2


—

network bound
gpu: A B C
10 GB for the given batch size
network bound: 100 tokens/sec bottleneck
there is case where prefill throughput is lower than 100 t/s

net: 100
A: 200 prefill -> 80 
B: 250 prefill

PP=2: [A] -> [network] -> [B]
for prefill, 


A: 400 t/s -> gpu util will be low
B: 300
C: 300

—

Ax4, Bx4

PP=2, A TP=4, [B TP=2]x2

—

1. approach
one chain output: [V100x4,TP]->[A100]->[T4, TPx8]

current gpu pool = previous gpu pool - used ones

---
2. approach
one chain output: ([V100x4,TP]->[A100]->[T4, TPx8]) x X number of DP as much as possible

---

cost constraint


let me use the example,

10 decoder layers for the llm model (layer id: 0,1,2,3,4,5,6,7,9)
four gpu machines of GPU type A, and it has $1/h and FLOP is 100
four gpu machines of GPU type B, and it has $0.5h and FLOP is 40
so A gpu type has generally better cost and FLOP tradeoff. (100/1 v.s. 40/0.5). generally it makes sense to prefer GPU type A. and if the model can fit in type A gpu machines, then as soon as we start to use GPU type B, the $/token will decrease…. correct? or am I missing something? there is a case where we cannot fit the llm model in available gpu type machine A only, then now it makes sense to use type B. otherwise, in my opinion, we don’t need to use gpu type B. what do you think? do you think my analysis is correct? think critically.

different case. we only two machines for type A and also type B. so total four machines.
let's say for some reason, the optimal TP degress is TP=1 since the network between machines is slow. now the solution is how many PP (PP degree) and which machine and how many layers per machine.
if the entire model can fit in two machines of A and B with PP=2. then we need to find the optimal way to split the model into two parts and put them in each gpu A and B for the best throughput. maybe A is higher-end than B so six layers to A (0-6 layers to A) and three layers to B (7-9 layers to B). okay good.
we also want to check if PP=3 is better than PP=2. PP=3with A->A->B and also possibly PP=3 with A->B->B. and we will compare against PP=2, A->B the above case. my question is considering the cost and FLOP of each machine type (GPU type A, and it has $1/h and FLOP is 100,  GPU type B, and it has $0.5h and FLOP is 40), will there be a case where PP=3 can be actually better $/token than PP=2….


---

## Mankeerat's google colab code. total memory consumption filtering and enumeration heuristic

```python
import boto3, botocore
import time, math, itertools, csv, sys
from datetime import datetime, timedelta, timezone

import os
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


# -------- settings --------
PRODUCT_DESC = ["Linux/UNIX"]
SPOT_LOOKBACK_MIN = 60
OUTPUT_CSV = "aws_gpu_spot_inventory.csv"
AWS_RETRY = 3
AWS_RETRY_SLEEP = 1.5
PAGE_SIZE = 100  # API hard limit for DescribeInstanceTypes
# --------------------------


def _retry(fn, *args, **kwargs):
    for i in range(AWS_RETRY):
        try:
            return fn(*args, **kwargs)
        except botocore.exceptions.ClientError as e:
            if i == AWS_RETRY - 1:
                raise
            time.sleep(AWS_RETRY_SLEEP * (i + 1))

def get_all_regions():
    # seed with a bootstrap region
    ec2 = boto3.client("ec2", region_name="us-east-1")
    resp = _retry(ec2.describe_regions, AllRegions=False)
    return sorted(r["RegionName"] for r in resp["Regions"])

def list_gpu_instance_types(region):
    """Return dict[it] = metadata for GPU-backed types supporting spot."""
    ec2 = boto3.client("ec2", region_name=region)
    details = {}
    paginator = ec2.get_paginator("describe_instance_types")
    for page in paginator.paginate():
        for it in page.get("InstanceTypes", []):
            # Must have GPU
            gi = it.get("GpuInfo")
            if not gi:
                continue
            # Must support Spot
            if "SupportedUsageClasses" in it and "spot" not in it["SupportedUsageClasses"]:
                continue

            itype = it["InstanceType"]
            gpus = gi.get("Gpus", [])
            gpu_count = sum(g.get("Count", 0) for g in gpus) if gpus else 0
            gpu_model = gpus[0].get("Name") if gpus else None
            gpu_mem_mib_each = gpus[0].get("MemoryInfo", {}).get("SizeInMiB") if gpus else None
            total_gpu_mem_mib = gpu_count * gpu_mem_mib_each if (gpu_count and gpu_mem_mib_each) else None

            vcpu = it.get("VCpuInfo", {}).get("DefaultVCpus")
            memory_mib = it.get("MemoryInfo", {}).get("SizeInMiB")
            net_perf = it.get("NetworkInfo", {}).get("NetworkPerformance")
            storage = it.get("InstanceStorageInfo")
            local_nvme = f"{storage['TotalSizeInGB']} GB" if storage and storage.get("TotalSizeInGB") else ""
            family = itype.split(".")[0]

            details[itype] = {
                "family": family,
                "gpu_model": gpu_model,
                "gpu_count": gpu_count,
                "gpu_mem_mib_each": gpu_mem_mib_each,
                "total_gpu_mem_mib": total_gpu_mem_mib,
                "vcpu": vcpu,
                "memory_mib": memory_mib,
                "network_performance": net_perf,
                "local_nvme": local_nvme,
            }
    return details

def get_spot_prices_last_hour(region, instance_types):
    """Return mapping itype -> (last_price, avg_price, count)."""
    if not instance_types:
        return {}
    ec2 = boto3.client("ec2", region_name=region)
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=SPOT_LOOKBACK_MIN)

    prices = {it: {"last": None, "sum": 0.0, "count": 0} for it in instance_types}
    paginator = ec2.get_paginator("describe_spot_price_history")
    for page in paginator.paginate(StartTime=start, EndTime=end, ProductDescriptions=PRODUCT_DESC):
        for rec in page.get("SpotPriceHistory", []):
            itype = rec["InstanceType"]
            if itype not in prices:
                continue
            price = float(rec["SpotPrice"])
            ts = rec["Timestamp"]
            if prices[itype]["last"] is None or ts > prices[itype]["last"][0]:
                prices[itype]["last"] = (ts, price)
            prices[itype]["sum"] += price
            prices[itype]["count"] += 1

    out = {}
    for itype in instance_types:
        d = prices[itype]
        last_price = d["last"][1] if d["last"] else None
        avg_price = (d["sum"] / d["count"]) if d["count"] else None
        out[itype] = (last_price, avg_price, d["count"])
    return out

def main():
    regions = get_all_regions()
    rows = []
    for region in regions:
        try:
            meta = list_gpu_instance_types(region)
            if not meta:
                continue
            spot = get_spot_prices_last_hour(region, list(meta.keys()))
            # Option: keep only types with any recent spot price samples
            for itype, m in meta.items():
                sp = spot.get(itype, (None, None, 0))
                if sp[2] == 0:
                    # No spot trading last 60m; skip or include with nulls.
                    continue
                rows.append({
                    "region": region,
                    "instance_type": itype,
                    "family": m["family"],
                    "gpu_model": m["gpu_model"],
                    "gpu_count": m["gpu_count"],
                    "gpu_mem_mib_each": m["gpu_mem_mib_each"],
                    "total_gpu_mem_mib": m["total_gpu_mem_mib"],
                    "vcpu": m["vcpu"],
                    "memory_mib": m["memory_mib"],
                    "network_performance": m["network_performance"],
                    "local_nvme": m["local_nvme"],
                    "availability_zones": "",          # (optional) can add later
                    "offering_scope": "region",        # (we’re not using the offerings API)
                    "spot_price_last": sp[0],
                    "spot_price_avg_1h": sp[1],
                    "spot_samples_1h": sp[2],
                })
            print(f"[{region}] rows so far: {len(rows)}")
        except Exception as e:
            print(f"[WARN] Region {region} failed: {e}", file=sys.stderr)

    fieldnames = [
        "region","instance_type","family","gpu_model","gpu_count",
        "gpu_mem_mib_each","total_gpu_mem_mib","vcpu","memory_mib",
        "network_performance","local_nvme","availability_zones",
        "offering_scope","spot_price_last","spot_price_avg_1h","spot_samples_1h"
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {OUTPUT_CSV}")

if __name__ == "__main__":
    # Bootstrap region for clients that require one
    import os
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    main()
import itertools
import pandas as pd
from typing import List, Tuple, Optional

path = "aws_gpu_spot_inventory.csv"
df = pd.read_csv(path)
# numeric cleanup
for c in ("spot_price_last", "spot_price_avg_1h", "total_gpu_mem_mib", "gpu_count"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# derive effective price and VRAM (GB)
def _effective_price(row) -> float:
    return (row.get("spot_price_last")
            if pd.notnull(row.get("spot_price_last"))
            else row.get("spot_price_avg_1h"))

df["spot_price_effective"] = df.apply(_effective_price, axis=1)
df["total_gpu_mem_gb"] = (df["total_gpu_mem_mib"] / 1024.0)

# keep only rows with price & non-zero VRAM & at least 1 GPU
df = df.dropna(subset=["spot_price_effective", "total_gpu_mem_gb", "gpu_count"])
df = df[(df["total_gpu_mem_gb"] > 0) & (df["gpu_count"] > 0)]

# optional accelerator filter

# collapse duplicates: keep cheapest (region, instance_type)
df = df.sort_values("spot_price_effective", ascending=True)
df = df.groupby(["region", "instance_type"], as_index=False).first()

# select usable columns
use_cols = [
    "region", "instance_type", "gpu_model",
    "total_gpu_mem_gb", "spot_price_effective"
]
final = df[use_cols].rename(columns={
    "total_gpu_mem_gb": "vram_gb",
    "spot_price_effective": "hourly_cost"
})

#region wise only

REGION_FILTER = ["us-east-1"]  # or None for all regions

import os, re, json, sys, time
import pandas as pd
import numpy as np
from itertools import permutations
from datetime import datetime

# ------------------- KNOBS -------------------
MAX_INSTANCES = 5
BUCKET_SIZE_GB = 4
TARGET_MIN_GB = 500
TARGET_MAX_GB = 700
TARGET_STEP_GB = 4
EXACT = True          # if False, accept >= target up to target+OVERFILL_GB
OVERFILL_GB = 0
MAX_RESULTS_PER_TARGET = None     # cap enumeration per target (None = uncapped)

# REGION_FILTER = None              # e.g., ["us-east-1","us-west-2"] or None for all
FRONTIER_K = 5               # e.g., 3 or 5 to keep K cheapest per mem_bucket inside each region

# NVIDIA families (case-insensitive)
NVIDIA_WHITELIST = ["T4","T4G","A10G","L4","L40","L40S","V100","A100","H100","H200","B200","RTX","TESLA"]

# Decode compute ms (swap with your benches)
GPU_COMPUTE_MS = {
    "T4": 22.0, "T4G": 22.0, "A10G": 6.0,
    "L4": 9.0, "L40": 6.5, "L40S": 5.8,
    "V100": 13.5, "A100": 5.0, "H100": 3.2, "H200": 2.6, "B200": 2.3,
    "RTX": 10.0, "TESLA": 14.0,
}

# Region latency estimates (ms). For region-only, this will usually collapse to ~2ms per hop.
DEFAULT_INTERREGION_MS = 120.0
DEFAULT_INTRAREGION_MS = 2.0
REGION_LAT_MS = {
    ("us-east-1","us-east-1"): 1.0,
    ("us-west-1","us-west-1"): 2.0,
    ("us-west-2","us-west-2"): 2.0,
    ("eu-west-1","eu-west-1"): 2.0,
    ("ap-southeast-1","ap-southeast-1"): 2.0,
    ("us-east-1","us-west-1"): 70.0, ("us-west-1","us-east-1"): 70.0,
    ("us-east-1","us-west-2"): 65.0, ("us-west-2","us-east-1"): 65.0,
    ("us-east-1","eu-west-1"): 70.0, ("eu-west-1","us-east-1"): 70.0,
    ("us-east-1","ap-southeast-1"): 180.0, ("ap-southeast-1","us-east-1"): 180.0,
}
# ---------------------------------------------

def find_csv():
    candidates = []
    for base in [".", "/mnt/data"]:
        if os.path.isdir(base):
            for name in os.listdir(base):
                if re.match(r"aws_gpu_spot_inventory.*\.csv$", name, re.I):
                    candidates.append(os.path.join(base, name))
    if not candidates:
        raise FileNotFoundError("Couldn't find a file like 'aws_gpu_spot_inventory*.csv' in . or /mnt/data")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: re.sub(r"[^\w]+","_", c.strip().lower()) for c in df.columns}
    return df.rename(columns=mapping)

def choose_col(df: pd.DataFrame, options: list[str]) -> str | None:
    for opt in options:
        if opt in df.columns:
            return opt
    return None

def ci_contains(s: str, needles) -> bool:
    if not isinstance(s, str): return False
    u = s.upper()
    return any(tok in u for tok in needles)

def load_inventory(path: str, bucket_gb: int) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = normalize_cols(raw)

    # numeric coerce
    for c in ["spot_price_last","spot_price_avg_1h","spot_price_effective","total_gpu_mem_mib","gpu_count","vram_gb"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # hourly_cost
    if choose_col(df, ["spot_price_effective"]):
        hourly = df["spot_price_effective"]; price_src = "spot_price_effective"
    else:
        last = df[choose_col(df, ["spot_price_last"])] if choose_col(df, ["spot_price_last"]) else pd.Series(np.nan, index=df.index)
        avg  = df[choose_col(df, ["spot_price_avg_1h"])] if choose_col(df, ["spot_price_avg_1h"]) else pd.Series(np.nan, index=df.index)
        hourly = last.combine_first(avg); price_src = "spot_price_last/spot_price_avg_1h"
    if "hourly_cost" in df.columns:
        hourly2 = pd.to_numeric(df["hourly_cost"], errors="coerce")
        hourly = hourly2.combine_first(hourly)
    df["hourly_cost"] = pd.to_numeric(hourly, errors="coerce")

    # vram in GB
    if "total_gpu_mem_mib" in df.columns:
        df["vram_gb"] = pd.to_numeric(df["total_gpu_mem_mib"], errors="coerce") / 1024.0
    elif "vram_gb" in df.columns:
        df["vram_gb"] = pd.to_numeric(df["vram_gb"], errors="coerce")
    else:
        raise ValueError("CSV must have total_gpu_mem_mib or vram_gb")

    need_cols = ["hourly_cost","vram_gb","gpu_count","gpu_model","region","instance_type"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}\nAvailable: {list(df.columns)}")

    # clean BEFORE bucketing
    df = df.dropna(subset=["hourly_cost","vram_gb","gpu_count"])
    df = df[(df["vram_gb"] > 0) & (pd.to_numeric(df["gpu_count"], errors="coerce") > 0)]

    # NVIDIA-only
    df = df[df["gpu_model"].apply(lambda x: ci_contains(x, NVIDIA_WHITELIST))]

    # cheapest per (region, instance_type)
    df = df.sort_values(["region","instance_type","hourly_cost"], ascending=[True,True,True])
    df = df.drop_duplicates(subset=["region","instance_type"], keep="first")

    # uid + mem buckets
    df["uid"] = df["instance_type"] + "@" + df["region"]
    mem_bucket = np.floor(df["vram_gb"]/bucket_gb) * bucket_gb
    df["mem_bucket"] = pd.to_numeric(mem_bucket, errors="coerce")
    df = df.dropna(subset=["mem_bucket"])
    df["mem_bucket"] = df["mem_bucket"].astype(int)
    df = df[df["mem_bucket"] > 0]

    out = df[["uid","region","instance_type","gpu_model","vram_gb","mem_bucket","hourly_cost"]].copy()

    print(f"Loaded '{os.path.basename(path)}' | rows: {len(out)} | price source: {price_src}")
    if len(out) == 0:
        print("WARNING: inventory empty after filtering; adjust whitelist/columns.")
    print(out.head(5).to_string(index=False))
    return out

# region latency + TPS prediction
def hop_latency_ms(a: str, b: str) -> float:
    if a == b:
        return REGION_LAT_MS.get((a,b), DEFAULT_INTRAREGION_MS)
    return REGION_LAT_MS.get((a,b), DEFAULT_INTERREGION_MS)

def compute_ms_for_model(model: str) -> float:
    if not isinstance(model, str): return 10.0
    mu = model.upper()
    for k,v in GPU_COMPUTE_MS.items():
        if k in mu:
            return v
    return 10.0

def best_path_latency_ms(regions: list[str]) -> tuple[float, list[int]]:
    k = len(regions)
    if k <= 1: return 0.0, list(range(k))
    best = float("inf"); best_order = None
    idxs = list(range(k))
    for order in set(permutations(idxs)):
        hops = 0.0
        for i in range(k-1):
            r1, r2 = regions[order[i]], regions[order[i+1]]
            hops += hop_latency_ms(r1, r2)
        if hops < best:
            best, best_order = hops, list(order)
    return best, best_order

def tps_region_aware(chain_rows: list[dict]) -> tuple[float, float, str]:
    if not chain_rows: return 0.0, 0.0, ""
    compute_ms = max(compute_ms_for_model(r["gpu_model"]) for r in chain_rows)
    regs = [r["region"] for r in chain_rows]
    hop_ms, order = best_path_latency_ms(regs)
    if order is None: order = list(range(len(regs)))
    order_str = "->".join(regs[i] for i in order)
    tps = 1000.0 / (compute_ms + hop_ms)
    return round(tps, 1), round(hop_ms, 1), order_str

SAFE_COLS = [
    "target_vram_gb","chain_len","total_vram_gb_bucketed","total_vram_gb_real",
    "hourly_cost","tps_est","best_path_hop_ms","best_region_order",
    "instance_chain","gpu_models","per_node_vram_gb_bucketed","per_node_price_hr"
]

def enumerate_chains_for_target(items_df: pd.DataFrame,
                                target_gb: int,
                                max_instances: int = MAX_INSTANCES,
                                exact: bool = EXACT,
                                overfill_gb: int = OVERFILL_GB,
                                max_results: int | None = MAX_RESULTS_PER_TARGET) -> pd.DataFrame:
    req = ["mem_bucket","region","instance_type","gpu_model","vram_gb","hourly_cost"]
    missing = [c for c in req if c not in items_df.columns]
    if missing:
        raise KeyError(f"items_df missing columns: {missing}\nHave: {list(items_df.columns)}")

    df = items_df.reset_index(drop=True).copy()
    df["mem_bucket"] = pd.to_numeric(df["mem_bucket"], errors="coerce")
    df["hourly_cost"] = pd.to_numeric(df["hourly_cost"], errors="coerce")
    df = df.dropna(subset=["mem_bucket","hourly_cost"])
    df["mem_bucket"] = df["mem_bucket"].astype(int)

    # optional within-region frontier: keep K cheapest per mem_bucket
    if FRONTIER_K is not None and FRONTIER_K > 0:
        df = (
            df.sort_values(["mem_bucket","hourly_cost"])
              .groupby("mem_bucket", as_index=False)
              .head(int(FRONTIER_K))
        )

    df = df.sort_values(["mem_bucket","hourly_cost"], ascending=[True, True])
    items = df.to_dict("records")
    mems  = [it["mem_bucket"] for it in items]
    costs = [float(it["hourly_cost"]) for it in items]
    n = len(items)

    rows, stack = [], []

    def dfs(start_idx, used, mem_sum, cost_sum):
        if used > 0:
            if exact:
                if mem_sum == target_gb:
                    chain = [items[i] for i in stack]
                    tps, hop_ms, order_str = tps_region_aware(chain)
                    rows.append((cost_sum, chain, tps, hop_ms, order_str))
            else:
                if target_gb <= mem_sum <= target_gb + overfill_gb:
                    chain = [items[i] for i in stack]
                    tps, hop_ms, order_str = tps_region_aware(chain)
                    rows.append((cost_sum, chain, tps, hop_ms, order_str))
                    if mem_sum >= target_gb:
                        return

        if used == max_instances:
            return

        for idx in range(start_idx, n):  # combinations-with-replacement
            new_mem = mem_sum + mems[idx]
            if exact:
                if new_mem > target_gb:
                    break
            else:
                if new_mem > target_gb + overfill_gb:
                    break
            stack.append(idx)
            dfs(idx, used+1, new_mem, cost_sum + costs[idx])
            stack.pop()
            if max_results is not None and len(rows) >= max_results:
                return

    dfs(0, 0, 0, 0.0)

    if not rows:
        return pd.DataFrame(columns=SAFE_COLS)

    out = []
    for cost, chain, tps, hop_ms, order_str in rows:
        out.append({
            "target_vram_gb": target_gb,
            "chain_len": len(chain),
            "total_vram_gb_bucketed": sum(x["mem_bucket"] for x in chain),
            "total_vram_gb_real": round(sum(x["vram_gb"] for x in chain), 1),
            "hourly_cost": round(cost, 4),
            "tps_est": tps,
            "best_path_hop_ms": hop_ms,
            "best_region_order": order_str,
            "instance_chain": "+".join(f'{x["instance_type"]}@{x["region"]}' for x in chain),
            "gpu_models": "+".join(x["gpu_model"] for x in chain),
            "per_node_vram_gb_bucketed": "+".join(str(x["mem_bucket"]) for x in chain),
            "per_node_price_hr": "+".join(f'{float(x["hourly_cost"]):.3f}' for x in chain),
        })
    df_out = pd.DataFrame(out)
    return df_out.sort_values(["target_vram_gb","hourly_cost","chain_len","total_vram_gb_bucketed"]).reset_index(drop=True)

# --------- PROGRESS + STREAMING CSV (REGION-ONLY) ---------
def maybe_tqdm(iterable, total=None, desc=""):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable

def append_csv(df: pd.DataFrame, path: str, header_cols=None):
    if df is None or df.empty:
        return 0
    header = not os.path.exists(path)
    if header and header_cols is not None:
        df = df.reindex(columns=header_cols)
    df.to_csv(path, mode="a", header=header, index=False)
    return len(df)

PROGRESS_PATH = "aws_chain_progress_regional.json"

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        try:
            with open(PROGRESS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_progress(obj):
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, PROGRESS_PATH)

def stream_regional(inv: pd.DataFrame, targets: list[int], out_path: str,
                    exact=True, overfill_gb=0, max_instances=5,
                    max_results_per_target=None, cols_order=None):
    prog = load_progress()
    done_pairs = set(tuple(x) for x in prog.get("regional_done_pairs", []))
    wrote_total = 0
    start = time.time()

    regions_all = sorted(inv["region"].unique().tolist())
    regions = [r for r in regions_all if (REGION_FILTER is None or r in REGION_FILTER)]
    print(f"[REGIONAL] writing -> {out_path} | regions={len(regions)} | resume pairs={len(done_pairs)}"
          + ("" if REGION_FILTER is None else f" | filter={REGION_FILTER}"))

    for region in regions:
        sub = inv[inv["region"] == region]
        # (Optional) apply FRONTIER_K within region before iterating targets
        if FRONTIER_K is not None and FRONTIER_K > 0:
            sub = (sub.sort_values(["mem_bucket","hourly_cost"])
                       .groupby("mem_bucket", as_index=False)
                       .head(int(FRONTIER_K)))
        for T in maybe_tqdm(targets, total=len(targets), desc=f"{region} targets"):
            key = (region, T)
            if key in done_pairs:
                continue
            df_t = enumerate_chains_for_target(
                sub, target_gb=T,
                max_instances=max_instances,
                exact=exact, overfill_gb=overfill_gb,
                max_results=max_results_per_target
            )
            if df_t is not None and not df_t.empty:
                df_t.insert(0, "region_scope", region)
                n = append_csv(df_t, out_path, header_cols=(["region_scope"] + SAFE_COLS))
                wrote_total += n
            # mark done even if empty
            done_pairs.add(key)
            prog["regional_done_pairs"] = sorted(list(done_pairs))
            prog["regional_out_path"] = out_path
            prog["updated_at"] = datetime.utcnow().isoformat() + "Z"
            save_progress(prog)
            if "tqdm" not in sys.modules:
                print(f"  {region} | T={T:4d} GB | rows={0 if df_t is None else len(df_t)} | cumulative={wrote_total}")

    dur = time.time() - start
    print(f"[REGIONAL] done: rows={wrote_total} in {dur:.1f}s → {out_path}")

# ---------- RUN (REGION-ONLY) ----------
INPUT_CSV = find_csv()
inv = load_inventory(INPUT_CSV, BUCKET_SIZE_GB)

targets = list(range(TARGET_MIN_GB, TARGET_MAX_GB + 1, TARGET_STEP_GB))
ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
regional_out = f"aws_regional_chains_{ts}.csv"

stream_regional(inv, targets, out_path=regional_out,
                exact=EXACT, overfill_gb=OVERFILL_GB,
                max_instances=MAX_INSTANCES,
                max_results_per_target=MAX_RESULTS_PER_TARGET)

print("Outputs:")
print("  Regional CSV:", regional_out)
print("  Progress    :", "aws_chain_progress_regional.json")

# Quick preview
try:
    preview_regional = pd.read_csv(regional_out, nrows=20)
    print("\n[Preview] regional_chains_df (first 20 rows):")
    display(preview_regional)
except Exception as e:
    print("Preview failed:", e)


region_scope
target_vram_gb
chain_len
total_vram_gb_bucketed
total_vram_gb_real
hourly_cost
tps_est
best_path_hop_ms
best_region_order
instance_chain
gpu_models
per_node_vram_gb_bucketed
per_node_price_hr
0
us-east-1
500
5
500
507.7
10.6819
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.xlarge@us-east-1+g6e.2xlarge@us-east-1+g6....
T4g+L40S+L4+L40S+L40S
16+44+88+176+176
0.137+0.910+2.014+3.810+3.810
1
us-east-1
500
5
500
507.7
10.7309
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.xlarge@us-east-1+g6e.xlarge@us-east-1+g6.1...
T4g+L40S+L4+L40S+L40S
16+44+88+176+176
0.137+0.959+2.014+3.810+3.810
2
us-east-1
500
5
500
507.7
10.7604
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.xlarge@us-east-1+g6e.2xlarge@us-east-1+g5....
T4g+L40S+A10G+L40S+L40S
16+44+88+176+176
0.137+0.910+2.093+3.810+3.810
3
us-east-1
500
5
500
507.7
10.7693
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g4dn.xlarge@us-east-1+g6e.2xlarge@us-east-1+g6...
T4+L40S+L4+L40S+L40S
16+44+88+176+176
0.225+0.910+2.014+3.810+3.810
4
us-east-1
500
5
500
507.7
10.8094
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.xlarge@us-east-1+g6e.xlarge@us-east-1+g5.1...
T4g+L40S+A10G+L40S+L40S
16+44+88+176+176
0.137+0.959+2.093+3.810+3.810
5
us-east-1
500
5
500
507.7
10.8183
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g4dn.xlarge@us-east-1+g6e.xlarge@us-east-1+g6....
T4+L40S+L4+L40S+L40S
16+44+88+176+176
0.225+0.959+2.014+3.810+3.810
6
us-east-1
500
5
500
507.7
10.8351
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.2xlarge@us-east-1+g6e.2xlarge@us-east-1+g6...
T4g+L40S+L4+L40S+L40S
16+44+88+176+176
0.290+0.910+2.014+3.810+3.810
7
us-east-1
500
5
500
507.7
10.8478
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g4dn.xlarge@us-east-1+g6e.2xlarge@us-east-1+g5...
T4+L40S+A10G+L40S+L40S
16+44+88+176+176
0.225+0.910+2.093+3.810+3.810
8
us-east-1
500
5
500
507.7
10.8727
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.4xlarge@us-east-1+g6e.2xlarge@us-east-1+g6...
T4g+L40S+L4+L40S+L40S
16+44+88+176+176
0.328+0.910+2.014+3.810+3.810
9
us-east-1
500
5
500
507.7
10.8746
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.xlarge@us-east-1+g6e.2xlarge@us-east-1+g6....
T4g+L40S+L4+L40S+L40S
16+44+88+176+176
0.137+0.910+2.014+3.810+4.003
10
us-east-1
500
5
500
507.7
10.8841
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.2xlarge@us-east-1+g6e.xlarge@us-east-1+g6....
T4g+L40S+L4+L40S+L40S
16+44+88+176+176
0.290+0.959+2.014+3.810+3.810
11
us-east-1
500
5
500
507.7
10.8886
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g4dn.2xlarge@us-east-1+g6e.2xlarge@us-east-1+g...
T4+L40S+L4+L40S+L40S
16+44+88+176+176
0.344+0.910+2.014+3.810+3.810
12
us-east-1
500
5
500
507.7
10.8968
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g4dn.xlarge@us-east-1+g6e.xlarge@us-east-1+g5....
T4+L40S+A10G+L40S+L40S
16+44+88+176+176
0.225+0.959+2.093+3.810+3.810
13
us-east-1
500
5
500
507.7
10.9136
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.2xlarge@us-east-1+g6e.2xlarge@us-east-1+g5...
T4g+L40S+A10G+L40S+L40S
16+44+88+176+176
0.290+0.910+2.093+3.810+3.810
14
us-east-1
500
5
500
507.7
10.9217
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.4xlarge@us-east-1+g6e.xlarge@us-east-1+g6....
T4g+L40S+L4+L40S+L40S
16+44+88+176+176
0.328+0.959+2.014+3.810+3.810
15
us-east-1
500
5
500
507.7
10.9236
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.xlarge@us-east-1+g6e.xlarge@us-east-1+g6.1...
T4g+L40S+L4+L40S+L40S
16+44+88+176+176
0.137+0.959+2.014+3.810+4.003
16
us-east-1
500
5
500
507.7
10.9376
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g4dn.2xlarge@us-east-1+g6e.xlarge@us-east-1+g6...
T4+L40S+L4+L40S+L40S
16+44+88+176+176
0.344+0.959+2.014+3.810+3.810
17
us-east-1
500
5
500
507.7
10.9512
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.4xlarge@us-east-1+g6e.2xlarge@us-east-1+g5...
T4g+L40S+A10G+L40S+L40S
16+44+88+176+176
0.328+0.910+2.093+3.810+3.810
18
us-east-1
500
5
500
507.7
10.9531
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.xlarge@us-east-1+g6e.2xlarge@us-east-1+g5....
T4g+L40S+A10G+L40S+L40S
16+44+88+176+176
0.137+0.910+2.093+3.810+4.003
19
us-east-1
500
5
500
507.7
10.9571
38.5
4.0
us-east-1->us-east-1->us-east-1->us-east-1->us...
g5g.xlarge@us-east-1+g6e.4xlarge@us-east-1+g6....
T4g+L40S+L4+L40S+L40S
16+44+88+176+176
0.137+1.185+2.014+3.810+3.810


result = pick_best_chain(140.0)  # llama-3.3-70b layers≈140 GB → total≈200 GB
print(result)
```