import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import subprocess
from torch.profiler import profile, ProfilerActivity

def profile_model_performance(model, model_name, model_size_mb, dataloader, device, num_batches=50, warmup_batches=10):
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    model.to(device)
    model.eval()
    latencies, peak_mems, avg_mems = [], [], []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= warmup_batches:
                break
            _ = model(inputs.to(device, dtype=model_dtype))
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device, dtype=model_dtype)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            with profile(activities=[ProfilerActivity.CUDA], profile_memory=True) as prof:
                _ = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
            if device == "cuda":
                peak_mems.append(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
                avg_mems.append(torch.cuda.memory_allocated(device) / (1024 * 1024))
    avg_latency = np.mean(latencies) if latencies else 0
    total_time_sec = np.sum(latencies) / 1000
    total_images = min(num_batches, len(dataloader)) * dataloader.batch_size
    throughput = total_images / total_time_sec if total_time_sec > 0 else 0
    avg_peak_mem = np.mean(peak_mems) if peak_mems else 0
    avg_used_mem = np.mean(avg_mems) if avg_mems else 0
    energy_mJ = None
    if device == "cuda":
        powers, batch_times = [], []
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                inputs = inputs.to(device, dtype=model_dtype)
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(inputs)
                torch.cuda.synchronize()
                end = time.perf_counter()
                batch_times.append(end - start)
                try:
                    device_id = torch.cuda.current_device()
                    output = subprocess.check_output(
                        f"nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i {device_id}",
                        shell=True
                    ).decode("utf-8").strip()
                    powers.append(float(output))
                except Exception:
                    pass
        if powers and batch_times:
            avg_power = np.mean(powers)
            total_time = np.sum(batch_times)
            energy_mJ = avg_power * total_time * 1000
    results = {
        "Model": model_name,
        "Size (MB)": f"{model_size_mb:.2f}",
        "Avg Latency (ms/batch)": f"{avg_latency:.2f}",
        "Throughput (img/sec)": f"{throughput:.2f}",
        # "Peak GPU Memory (MB)": f"{avg_peak_mem:.2f}" if device == "cuda" else "N/A",
        # "Avg GPU Memory (MB)": f"{avg_used_mem:.2f}" if device == "cuda" else "N/A",
        # "Energy (mJ)": f"{energy_mJ:.2f}" if energy_mJ is not None else "Unavailable",
    }
    return results
