#!/usr/bin/env python3
import argparse
import gc
import itertools
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stemlm.model import JSDMConfig, JSDMForMaskedSpeciesPrediction  # noqa: E402

NUM_SITES = 1000


def random_batch(config, batch_size, device):
    S, N, E = config.num_species, config.num_source_sites, config.num_env_vars
    return {
        "input_ids": torch.full((batch_size, S, 1), 2, device=device),
        "source_ids": torch.randint(0, 2, (batch_size, S, N), dtype=torch.uint8, device=device),
        "source_idx": torch.randint(0, NUM_SITES, (batch_size, N), device=device),
        "target_site_idx": torch.randint(0, NUM_SITES, (batch_size, 1), device=device),
        "env_data": torch.randn(batch_size, N, E, device=device),
        "target_env": torch.randn(batch_size, E, device=device),
        "site_lats": torch.empty(NUM_SITES, device=device).uniform_(25.0, 50.0),
        "site_lons": torch.empty(NUM_SITES, device=device).uniform_(-125.0, -65.0),
        "site_times": torch.empty(NUM_SITES, device=device).uniform_(0.0, 3650.0),
    }


def measure(config, batch_size, mode, device, repeats):
    model = JSDMForMaskedSpeciesPrediction(config).to(device)
    data = random_batch(config, batch_size, device)
    if mode == "training":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        labels = torch.randint(0, 2, (batch_size, config.num_species, 1), device=device)
    else:
        model.eval()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "training":
                loss = model(labels=labels, loss_type="focal", **data).loss
            else:
                with torch.no_grad():
                    model(**data)
        if mode == "training":
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    step()
    started = time.monotonic()
    for _ in range(repeats):
        step()
    seconds = (time.monotonic() - started) / repeats
    return seconds, torch.cuda.max_memory_allocated(device) / 2**20, sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(
        description="Time and peak GPU memory of one STEM-LM step on random inputs, by number of species "
                    "and sources. The batch size is halved until the step fits in GPU memory.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num_species", type=int, nargs="+", default=[171, 1201, 5000, 20000, 50000])
    parser.add_argument("--num_source_sites", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--num_env_vars", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    rows = []
    for S, N, mode in itertools.product(args.num_species, args.num_source_sites, ("inference", "training")):
        config = JSDMConfig(num_species=S, num_source_sites=N, num_env_vars=args.num_env_vars,
                            max_spatial_dist=5000.0, max_temporal_dist=3650.0,
                            temporal_fire_init_periods=(365.0, 182.0))
        batch_size = args.batch_size
        row = {"num_species": S, "num_source_sites": N, "mode": mode}
        while True:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                seconds, peak, num_params = measure(config, batch_size, mode, device, args.repeats)
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    row.update(batch_size=1, status="out of memory")
                    break
                batch_size //= 2
                continue
            row.update(batch_size=batch_size, status="ok", seconds_per_step=seconds,
                       sites_per_second=batch_size / seconds, peak_gpu_mb=round(peak), num_params=num_params)
            break
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
