#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import math
import time
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelWithLMHead
from huggingface_hub import InferenceClient

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# In[2]:


@dataclass
class Node:
    name: str
    device_type: str           # e.g. "NVIDIA-L40S"
    device_count: int          # total GPUs on node
    used_devices: int = 0      # GPUs currently allocated

    @property
    def free_devices(self) -> int:
        return self.device_count - self.used_devices


@dataclass
class Fabric:
    name: str
    nodes: List[Node]

    def total_free_by_type(self) -> Dict[str, int]:
        """Return total free GPUs per device_type."""
        totals: Dict[str, int] = {}
        for n in self.nodes:
            totals.setdefault(n.device_type, 0)
            totals[n.device_type] += n.free_devices
        return totals

    def alloc_on_type(
        self, device_type: str, gpus_needed: int
    ) -> Optional[List[Tuple[str, int]]]:
        """
        Try to allocate gpus_needed GPUs of device_type across nodes.
        Returns list of (node_name, num_gpus) or None if impossible.
        Simple greedy: fill most-free nodes first.
        """
        allocations: List[Tuple[str, int]] = []
        remaining = gpus_needed

        candidates = sorted(
            [n for n in self.nodes if n.device_type == device_type],
            key=lambda n: n.free_devices,
            reverse=True,
        )

        for node in candidates:
            if remaining <= 0:
                break
            take = min(node.free_devices, remaining)
            if take > 0:
                allocations.append((node.name, take))
                node.used_devices += take
                remaining -= take

        if remaining > 0:
            # rollback
            for name, num in allocations:
                for n in self.nodes:
                    if n.name == name:
                        n.used_devices -= num
                        break
            return None

        return allocations


# In[9]:


@dataclass
class JobSpec:
    job_id: str
    tenant_id: str
    model_name: str
    num_lines: int                # e.g. 4000, 2000
    avg_input_tokens: int         # estimated per line
    avg_output_tokens: int        # estimated per line
    slo_hours: float              # SLO from now, in hours
    job_type: str = "batch"       # "batch" or "online"
    importance: int = 1           # 1-low, 3-high


@dataclass
class JobState:
    spec: JobSpec
    submitted_at: float
    progress_frac: float = 0.0    # 0.0–1.0
    device_type: Optional[str] = None
    tp: int = 0
    pp: int = 0
    replicas: int = 0
    allocated_gpus: int = 0
    allocations: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def deadline_ts(self) -> float:
        return self.submitted_at + self.spec.slo_hours * 3600.0

    @property
    def total_tokens(self) -> int:
        return self.spec.num_lines * (self.spec.avg_input_tokens + self.spec.avg_output_tokens)

    @property
    def remaining_tokens(self) -> int:
        return int((1.0 - self.progress_frac) * self.total_tokens)


@dataclass
class PerfEntry:
    model_name: str
    device_type: str
    tp: int
    pp: int
    tokens_per_sec: float   # per replica (tp*pp GPUs)
    mem_per_gpu_gb: float


class PerfDB:
    def __init__(self, entries: List[PerfEntry]):
        self.entries = entries

    def lookup(self, model_name: str, device_type: str, tp: int, pp: int) -> Optional[PerfEntry]:
        for e in self.entries:
            if (
                e.model_name == model_name
                and e.device_type == device_type
                and e.tp == tp
                and e.pp == pp
            ):
                return e
        return None


# In[18]:


# Load your perfdb_l40s.csv
# perf_df = pd.read_csv("perfdb_l40s.csv")
perf_df = pd.read_csv("gangmuk_perfdb.csv")
print("Perf DB head:")
display(perf_df.head())

# Adjust these column names if your CSV headers differ.
entries = [
    PerfEntry(
        model_name=row["model_name"],
        device_type=row["device_type"],
        tp=int(row["tp"]),
        pp=int(row["pp"]),
        tokens_per_sec=float(row["tokens_per_sec"]),
        mem_per_gpu_gb=float(row["mem_per_gpu_gb"]),
    )
    for _, row in perf_df.iterrows()
]

perf_db = PerfDB(entries)

# Deduce device type from DB
default_device_type = perf_df["device_type"].iloc[0]
print("Default device_type from perfdb:", default_device_type)


# In[19]:


def llm_choose_config_from_candidates(
    job: JobState,
    candidates: List[Dict[str, Any]],
    model_id: str,
    hf_token: str,
    advisor_name: str,
    top_k: int = 3,
    temperature: float = 0.2,
) -> Optional[Dict[str, Any]]:
    """
    Ask an HF LLM (Phi or other) to choose among top_k analytic candidates.
    Returns the chosen config dict or None.
    """
    if not candidates:
        return None

    candidates_sorted = sorted(candidates, key=lambda c: c["gpu_time"])
    candidates_top = candidates_sorted[: min(top_k, len(candidates_sorted))]

    labels = ["A", "B", "C", "D", "E"]
    labeled = list(zip(labels, candidates_top))

    client = InferenceClient(model=model_id, token=hf_token)

    prompt_lines = []
    prompt_lines.append(
        f"You are an expert GPU scheduler ({advisor_name}) choosing tensor/pipeline "
        "parallelism for an LLM job.\n"
    )
    prompt_lines.append("Goals, in order:\n")
    prompt_lines.append("1. The job must finish within its SLO (deadline).\n")
    prompt_lines.append("2. Minimize total GPU-hours used.\n")
    prompt_lines.append("3. Prefer simpler configs (fewer TP/PP/replicas) when close.\n\n")

    prompt_lines.append("Job:\n")
    prompt_lines.append(f"- Job ID: {job.spec.job_id}\n")
    prompt_lines.append(f"- Model: {job.spec.model_name}\n")
    prompt_lines.append(f"- Lines (requests): {job.spec.num_lines}\n")
    prompt_lines.append(f"- Avg input tokens: {job.spec.avg_input_tokens}\n")
    prompt_lines.append(f"- Avg output tokens: {job.spec.avg_output_tokens}\n")
    prompt_lines.append(f"- SLO: {job.spec.slo_hours} hours\n")
    prompt_lines.append(f"- Total tokens (approx): {job.total_tokens}\n\n")

    prompt_lines.append("Candidate configs:\n")
    for label, cfg in labeled:
        prompt_lines.append(
            f"Plan {label}:\n"
            f"- tp: {cfg['tp']}\n"
            f"- pp: {cfg['pp']}\n"
            f"- replicas: {cfg['replicas']}\n"
            f"- total GPUs: {cfg['gpus_needed']}\n"
            f"- predicted runtime: {cfg['runtime_hours']:.2f} hours\n"
            f"- GPU-hours: {cfg['gpu_time']:.2f}\n\n"
        )

    prompt_lines.append(
        "Which plan best satisfies the goals? Respond with exactly one line:\n"
        "Best plan: A\n"
    )

    prompt = "".join(prompt_lines)

    try:
        resp = client.text_generation(
            prompt,
            max_new_tokens=64,
            temperature=temperature,
            do_sample=False,
        )
        text = resp.strip()
    except Exception as e:
        print(f"[LLM {advisor_name}] Error calling HF model: {e}")
        return None

    chosen_label = None
    for label, _ in labeled:
        if f"Best plan: {label}" in text or f"best plan: {label}" in text:
            chosen_label = label
            break
        if f"Plan {label}" in text or f"plan {label}" in text:
            chosen_label = label
            break

    if chosen_label is None:
        # fallback
        for label, _ in labeled:
            if f" {label}" in text:
                chosen_label = label
                break

    if chosen_label is None:
        print(f"[LLM {advisor_name}] Could not parse choice from: {text}")
        return None

    for label, cfg in labeled:
        if label == chosen_label:
            return cfg

    return None


# In[20]:


def load_c_pmi_model(model_name: str = "microsoft/DialoGPT-large"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def _avg_nll(text: str, tokenizer, model) -> float:
    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs[0]
    return loss.item()


def c_pmi_score(
    context: str,
    hypothesis: str,
    tokenizer,
    model,
    sep: str = " <|endoftext|> ",
) -> float:
    lpx = -_avg_nll(context + sep + hypothesis, tokenizer, model)
    lpx_context = -_avg_nll(context, tokenizer, model)
    lpx_hyp = -_avg_nll(hypothesis, tokenizer, model)
    pmi = lpx - lpx_context - lpx_hyp
    return pmi


def c_pmi_rank_plans(
    context: str,
    plan_labels,
    tokenizer,
    model,
    temperature: float = 1.0,
):
    hypotheses = [
        f"In this situation, the best plan is {label}."
        for label in plan_labels
    ]
    scores = [
        c_pmi_score(context, hyp, tokenizer, model)
        for hyp in hypotheses
    ]
    max_s = max(scores)
    exps = [math.exp((s - max_s) / max(temperature, 1e-6)) for s in scores]
    Z = sum(exps)
    probs = [e / Z for e in exps]
    label_to_prob = {label: prob for label, prob in zip(plan_labels, probs)}
    best_label = max(label_to_prob.items(), key=lambda x: x[1])[0]
    return best_label, label_to_prob


# Load the C-PMI scorer model
c_pmi_model, c_pmi_tokenizer = load_c_pmi_model()


# In[21]:


class OrcaOrchestrator:
    def __init__(
        self,
        fabric: Fabric,
        perf_db: PerfDB,
        hf_token: Optional[str] = None,
        hf_phi_model_id: Optional[str] = None,
        hf_other_model_id: Optional[str] = None,
    ):
        self.fabric = fabric
        self.perf_db = perf_db
        self.jobs: Dict[str, JobState] = {}

        self.hf_token = hf_token
        self.hf_phi_model_id = hf_phi_model_id
        self.hf_other_model_id = hf_other_model_id

    def submit_job(self, spec: JobSpec) -> JobState:
        now = time.time()
        job_state = JobState(spec=spec, submitted_at=now)
        self.jobs[spec.job_id] = job_state
        self._schedule_new_job(job_state, now)
        return job_state

    def _schedule_new_job(self, job: JobState, now: float):
        totals = self.fabric.total_free_by_type()
        if not totals:
            print(f"[Orca] No free GPUs for job {job.spec.job_id}")
            return

        device_type = next(iter(totals.keys()))
        free_gpus = totals[device_type]

        candidates = self._enumerate_configs(job, device_type, free_gpus, now)
        if not candidates:
            print(f"[Orca] Cannot meet SLO for job {job.spec.job_id} with free GPUs")
            return

        # --- 1) Math advisor: analytic best (minimal GPU-hours) ---
        math_cfg = min(candidates, key=lambda c: c["gpu_time"])

        # --- 2) Phi advisor ---
        phi_cfg = None
        if self.hf_token and self.hf_phi_model_id and len(candidates) > 1:
            phi_cfg = llm_choose_config_from_candidates(
                job=job,
                candidates=candidates,
                model_id=self.hf_phi_model_id,
                hf_token=self.hf_token,
                advisor_name="PhiAdvisor",
                top_k=3,
            )

        # --- 3) Other reasoning advisor ---
        other_cfg = None
        if self.hf_token and self.hf_other_model_id and len(candidates) > 1:
            other_cfg = llm_choose_config_from_candidates(
                job=job,
                candidates=candidates,
                model_id=self.hf_other_model_id,
                hf_token=self.hf_token,
                advisor_name="OtherAdvisor",
                top_k=3,
            )

        # --- Build context string for C-PMI judge ---
        plans = [("Math", math_cfg)]
        if phi_cfg is not None:
            plans.append(("PhiAdvisor", phi_cfg))
        if other_cfg is not None:
            plans.append(("OtherAdvisor", other_cfg))

        context_lines = []
        context_lines.append(
            "We are choosing a GPU parallelism configuration for an LLM job.\n"
        )
        context_lines.append("Goals:\n")
        context_lines.append("1. Meet the SLO (deadline).\n")
        context_lines.append("2. Minimize total GPU-hours.\n")
        context_lines.append("3. Prefer simpler configs when close.\n\n")

        context_lines.append("Job:\n")
        context_lines.append(f"- Model: {job.spec.model_name}\n")
        context_lines.append(f"- Lines: {job.spec.num_lines}\n")
        context_lines.append(f"- Avg input tokens: {job.spec.avg_input_tokens}\n")
        context_lines.append(f"- Avg output tokens: {job.spec.avg_output_tokens}\n")
        context_lines.append(f"- SLO: {job.spec.slo_hours} hours\n")
        context_lines.append(f"- Total tokens: {job.total_tokens}\n\n")

        context_lines.append("Advisor proposals:\n")
        for name, cfg in plans:
            context_lines.append(
                f"{name} proposes:\n"
                f"  - tp: {cfg['tp']}\n"
                f"  - pp: {cfg['pp']}\n"
                f"  - replicas: {cfg['replicas']}\n"
                f"  - total GPUs: {cfg['gpus_needed']}\n"
                f"  - predicted runtime: {cfg['runtime_hours']:.2f} hours\n"
                f"  - GPU-hours: {cfg['gpu_time']:.2f}\n\n"
            )

        context_str = "".join(context_lines)
        plan_labels = [name for (name, _) in plans]

        # --- 4) C-PMI judge decides which advisor is most likely "correct" ---
        best_label, probs = c_pmi_rank_plans(
            context=context_str,
            plan_labels=plan_labels,
            tokenizer=c_pmi_tokenizer,
            model=c_pmi_model,
        )

        print("[C-PMI Judge] Probabilities:", probs)
        print("[C-PMI Judge] Chose:", best_label)

        chosen_cfg = math_cfg
        for name, cfg in plans:
            if name == best_label:
                chosen_cfg = cfg
                break

        # --- 5) Place chosen config on the fabric ---
        alloc = self.fabric.alloc_on_type(device_type, chosen_cfg["gpus_needed"])
        if alloc is None:
            print(f"[Orca] Placement failed for job {job.spec.job_id} (fragmentation)")
            return

        job.device_type = device_type
        job.tp = chosen_cfg["tp"]
        job.pp = chosen_cfg["pp"]
        job.replicas = chosen_cfg["replicas"]
        job.allocated_gpus = chosen_cfg["gpus_needed"]
        job.allocations = alloc

        print(
            f"[Orca] Scheduled {job.spec.job_id} on {device_type}: "
            f"{chosen_cfg['gpus_needed']} GPUs (tp={chosen_cfg['tp']}, pp={chosen_cfg['pp']}, "
            f"replicas={chosen_cfg['replicas']}), predicted_runtime={chosen_cfg['runtime_hours']:.2f}h"
        )

    def _enumerate_configs(
        self,
        job: JobState,
        device_type: str,
        free_gpus_of_type: int,
        now: float,
        guard_frac: float = 0.1,
    ) -> List[Dict[str, Any]]:
        total_tokens = job.remaining_tokens
        T_left = job.deadline_ts - now
        if T_left <= 0:
            return []

        effective_horizon = T_left * (1.0 - guard_frac)
        if effective_horizon <= 0:
            return []

        configs: List[Dict[str, Any]] = []

        # You can expand TP/PP candidate sets later
        candidate_tps = [1, 2, 4]
        candidate_pps = [1]  #change/add em

        for tp in candidate_tps:
            for pp in candidate_pps:
                gpus_per_replica = tp * pp
                if gpus_per_replica > free_gpus_of_type:
                    continue

                pe = self.perf_db.lookup(job.spec.model_name, device_type, tp, pp)
                if pe is None:
                    continue

                # basic memory check; adjust if needed
                # here we assume device has at least pe.mem_per_gpu_gb
                max_replicas = free_gpus_of_type // gpus_per_replica
                if max_replicas == 0:
                    continue

                for replicas in range(1, max_replicas + 1):
                    tokens_per_sec_total = replicas * pe.tokens_per_sec
                    runtime_sec = total_tokens / tokens_per_sec_total

                    if runtime_sec <= effective_horizon:
                        gpu_count = replicas * gpus_per_replica
                        gpu_time = gpu_count * runtime_sec / 3600.0

                        configs.append({
                            "tp": tp,
                            "pp": pp,
                            "replicas": replicas,
                            "gpus_needed": gpu_count,
                            "runtime_hours": runtime_sec / 3600.0,
                            "gpu_time": gpu_time,
                        })
                        break  # minimal replicas that satisfy SLO for this tp/pp

        return configs


# In[22]:


# ----- Build a small L40S fabric -----
# We’ll use 4 nodes * 4 GPUs each for demo (16 GPUs total)
nodes = [
    Node(name=f"node-{i}", device_type=default_device_type, device_count=4)
    for i in range(4)
]
fabric = Fabric(name="l40s-fabric", nodes=nodes)

# ----- HF models for advisors -----
HF_TOKEN = "YOUR_HF_TOKEN_HERE"

HF_MODEL_PHI   = "microsoft/Phi-3-mini-4k-instruct"   # advisor 1
HF_MODEL_OTHER = "Qwen/Qwen2.5-1.5B-Instruct"         # advisor 2 (example)

orca = OrcaOrchestrator(
    fabric=fabric,
    perf_db=perf_db,
    hf_token=HF_TOKEN,
    hf_phi_model_id=HF_MODEL_PHI,
    hf_other_model_id=HF_MODEL_OTHER,
)

# ----- Example jobs -----

# Job 1: LLaMA 70B-like batch classification job, 12h SLO
job1_spec = JobSpec(
    job_id="job-llama-70b-batch",
    tenant_id="tenant-A",
    model_name="llama-3.3-70b",   # must match something in perfdb_l40s.csv
    num_lines=4000,
    avg_input_tokens=2048,
    avg_output_tokens=32,
    slo_hours=12.0,
    job_type="batch",
    importance=2,
)

# Job 2: DeepSeek distill 70B translation job, 6h SLO
job2_spec = JobSpec(
    job_id="job-deepseek-70b-batch",
    tenant_id="tenant-A",
    model_name="deepseek-distill-70b",  # must match perfdb
    num_lines=2000,
    avg_input_tokens=1024,
    avg_output_tokens=1024,
    slo_hours=6.0,
    job_type="batch",
    importance=2,
)

job1_state = orca.submit_job(job1_spec)
job2_state = orca.submit_job(job2_spec)

print("\n--- Job placements ---")
print(job1_state)
print(job2_state)

print("\n--- Fabric usage ---")
for n in fabric.nodes:
    print(n.name, "used/free:", n.used_devices, "/", n.device_count)


# In[ ]:




