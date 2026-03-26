import os
import re
import json
import sys
import csv
from statistics import mean, stdev
from pathlib import Path

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def get_nested(d, path):
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

def extract_power(payload):
    """Restituisce potenza in mW."""
    if payload is None: 
        return None
    rails = [
        "power.rail.VDD_GPU.power",
        "power.rail.VDD_CPU_GPU_CV.power",
        "power.tot.power"
    ]
    for r in rails:
        v = get_nested(payload, r)
        if isinstance(v, (int, float)):
            return float(v)
    return None

def extract_ram(payload):
    """RAM in MB."""
    if payload is None:
        return None
    v = get_nested(payload, "mem.RAM.used")
    if isinstance(v, (int, float)):
        return v / 1024.0
    return None

def extract_latencies(payload):
    """Estrae tutte le latenze in ms da un file di times."""
    if payload is None:
        return []
    vals = []
    if isinstance(payload, list):
        for e in payload:
            if isinstance(e, dict) and "latencyMs" in e:
                vals.append(float(e["latencyMs"]))
    if isinstance(payload, dict):
        if "latencyMs" in payload:
            vals.append(float(payload["latencyMs"]))
        for key in ("entries", "iterations", "data"):
            arr = payload.get(key)
            print(arr)
            if isinstance(arr, list):
                for e in arr:
                    if isinstance(e, dict) and "latencyMs" in e:
                        vals.append(float(e["latencyMs"]))
    return vals



RUN_LAST   = re.compile(r"^last_(?!times)(.+)\.json$")
RUN_QNET   = re.compile(r"^NN_(?!times)(.+)\.json$")

RUN_LAST_T = re.compile(r"^last_times(?:_(.+))?\.json$")
RUN_QNET_T = re.compile(r"^NN_times(?:_(.+))?\.json$")


def scan_runs(folder):
    runs = {}
    for f in os.listdir(folder):
        m = RUN_LAST.match(f)
        if m:
            rid = m.group(1)
            runs.setdefault(rid, {})["last"] = os.path.join(folder, f)
        m = RUN_QNET.match(f)
        if m:
            rid = m.group(1)
            runs.setdefault(rid, {})["NN"] = os.path.join(folder, f)
        m = RUN_LAST_T.match(f)
        if m:
            rid = m.group(1)
            runs.setdefault(rid, {})["last_t"] = os.path.join(folder, f)
        m = RUN_QNET_T.match(f)
        if m:
            rid = m.group(1)
            runs.setdefault(rid, {})["NN_t"] = os.path.join(folder, f)
    return runs

import os
from statistics import mean, stdev

def process_folder(folder):
    runs = scan_runs(folder)
    if not runs:
        return None

    power_list, ram_list = [], []

    # --- 1) POWER & RAM per-run ---
    for rid, paths in runs.items():
        # prendi i payload last/NN (se ci sono)
        last = load_json(paths.get("last"))
        NN = load_json(paths.get("NN"))

        # Somma power e ram dei due payload disponibili
        p = 0.0
        r = 0.0
        for payload in (last, NN):
            v = extract_power(payload)
            if isinstance(v, float):
                p += v
            rv = extract_ram(payload)
            if isinstance(rv, float):
                r += rv

        if p > 0.0:
            power_list.append(p)
            ram_list.append(r)

    if not power_list:
        return None

    if len(power_list) == 1:
        power_mean, power_std = power_list[0], 0.0
        ram_mean, ram_std = ram_list[0], 0.0
    else:
        power_mean, power_std = mean(power_list), stdev(power_list)
        ram_mean, ram_std = mean(ram_list), stdev(ram_list)
        
    times_paths = set()
    for _, paths in runs.items():
        if paths.get("last_t"):
            times_paths.add(paths["last_t"])
        if paths.get("NN_t"):
            times_paths.add(paths["NN_t"])

    latency_samples = []
    for tpath in times_paths:
        latency_samples += extract_latencies(load_json(tpath))

    if not latency_samples:
        return None

    if len(latency_samples) == 1:
        latency_mean, latency_std = latency_samples[0], 0.0
    else:
        latency_mean, latency_std = mean(latency_samples), stdev(latency_samples)

    energy_mean = power_mean * latency_mean 
    energy_std = None
    
    folder_name = os.path.basename(folder)

    return (
        folder_name,
        len(power_list),  
        power_mean, power_std,
        latency_mean, latency_std,
        ram_mean, ram_std,
        energy_mean, energy_std
    )
    
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("-o", default="out_report/metrics.csv")
    args = ap.parse_args()

    rows = []
    for d, dirs, files in os.walk(args.root):
        if any(re.match(r"(last|NN|last_times|NN_times)_", f) for f in files):
            res = process_folder(d)
            if res:
                rows.append(res)

    out_folder = "/".join(args.o.split('/')[:-1])
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(args.o):
        with open(args.o, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "folder", "runs",
                "power_mean_mW", "power_std_mW",
                "latency_mean_ms", "latency_std_ms",
                "ram_mean_MB", "ram_std_MB",
                "energy_mean_uJ", "energy_std_uJ"
            ])
    
    with open(args.o, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(rows[0])

    print(f"Done. Wrote {len(rows)} rows to {args.o}")


if __name__ == "__main__":
    main()