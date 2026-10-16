"""Execute the preregistered fixed-budget comparison; no test-based selection."""
import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import prepare
from .models import from_config
from .metrics import evaluate, generate, text_metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n")


def write_csv(path, records):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"run_manifest.json").exists():
        raise SystemExit("Refusing to overwrite existing evidence; use a new --output directory.")
    if config["device"] != "cpu":
        raise ValueError("This audited implementation uses CPU for deterministic comparable timings.")
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    source_paths = sorted(Path("src").glob("*.py")) + [Path(args.config)]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    environment = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                   "platform": platform.platform(), "processor": platform.processor(),
                   "device": "cpu", "threads": torch.get_num_threads(),
                   "mps_available": torch.backends.mps.is_available()}
    manifest = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "source_sha256": source_hashes, "environment": environment,
                "selection": "Lowest validation loss checkpoint within each run; model ranking by mean validation loss; primary generation seed is first configured training seed."}
    write_json(out/"run_manifest.json", manifest)
    datasets, vocab, origins, meta = prepare("data/raw/alice_pg11.txt", config)
    write_json(out/"dataset.json", meta)
    write_json(out/"vocab.json", vocab)
    np.savez_compressed(out/"sequence_origins.npz", **origins)
    (out/"checkpoints").mkdir(exist_ok=True)
    (out/"predictions").mkdir(exist_ok=True)
    loaders = {s: DataLoader(d, batch_size=config["batch_size"], shuffle=False) for s,d in datasets.items()}
    all_runs, histories = [], []
    for seed in config["seeds"]:
        # Same training order for each architecture, independently of initialization.
        for kind in config["models"]:
            run_start = time.perf_counter()
            set_seed(seed)
            model = from_config(kind, len(vocab), config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"])
            order_rng = torch.Generator().manual_seed(seed + 100000)
            train_loader = DataLoader(datasets["train"], batch_size=config["batch_size"],
                                      shuffle=True, generator=order_rng)
            best_loss, best_epoch, best_state = float("inf"), 0, None
            train_seconds = validation_seconds = 0.0
            for epoch in range(1, config["epochs"]+1):
                tick = time.perf_counter()
                model.train()
                loss_sum, count, correct = 0.0, 0, 0
                for x,y in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss: {kind}/{seed}/{epoch}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                    optimizer.step()
                    loss_sum += loss.item()*len(y)
                    correct += (logits.argmax(1)==y).sum().item()
                    count += len(y)
                elapsed = time.perf_counter()-tick
                train_seconds += elapsed
                tick = time.perf_counter()
                val, _ = evaluate(model, loaders["validation"])
                val_elapsed = time.perf_counter()-tick
                validation_seconds += val_elapsed
                histories.append({"model":kind,"seed":seed,"epoch":epoch,
                                  "train_online_loss":loss_sum/count,"train_online_accuracy":correct/count,
                                  "validation_loss":val["loss"],"validation_accuracy":val["accuracy"],
                                  "train_epoch_seconds":elapsed,"validation_seconds":val_elapsed})
                if val["loss"] < best_loss:
                    best_loss, best_epoch = val["loss"], epoch
                    best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
                print(f"{kind:5s} seed={seed} epoch={epoch:02d} train={loss_sum/count:.4f} val={val['loss']:.4f} seconds={elapsed:.2f}", flush=True)
            model.load_state_dict(best_state)
            torch.save({"model":kind,"seed":seed,"epoch":best_epoch,"config":config,
                        "vocab_size":len(vocab),"state_dict":best_state},
                       out/"checkpoints"/f"{kind}_seed{seed}.pt")
            record = {"model":kind,"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),
                      "best_epoch":best_epoch,"epochs_trained":config["epochs"],
                      "training_seconds":train_seconds,"validation_seconds":validation_seconds}
            for split in ["train","validation","test"]:
                tick = time.perf_counter()
                metrics, arrays = evaluate(model, loaders[split])
                record.update({f"{split}_{k}":v for k,v in metrics.items()})
                record[f"{split}_evaluation_seconds"] = time.perf_counter()-tick
                if split in ["validation", "test"]:
                    np.savez_compressed(out/"predictions"/f"{kind}_seed{seed}_{split}.npz", **arrays)
            record["run_wall_seconds"] = time.perf_counter()-run_start
            all_runs.append(record)
            write_csv(out/"runs.csv", all_runs)
            write_csv(out/"history.csv", histories)
    summaries = []
    keys = ["train_loss","validation_loss","test_loss","test_perplexity","test_accuracy",
            "test_known_word_accuracy","training_seconds","run_wall_seconds","best_epoch"]
    for kind in config["models"]:
        records = [r for r in all_runs if r["model"]==kind]
        summary = {"model":kind,"runs":len(records),"parameters":records[0]["parameters"]}
        for key in keys:
            values = np.asarray([r[key] for r in records], dtype=float)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_sd"] = float(values.std(ddof=1))
        summaries.append(summary)
    write_csv(out/"summary.csv", summaries)
    winner = min(summaries, key=lambda r:r["validation_loss_mean"])["model"]
    samples = []
    for kind in config["models"]:
        seed = config["seeds"][0]
        checkpoint = torch.load(out/"checkpoints"/f"{kind}_seed{seed}.pt", weights_only=True)
        model = from_config(kind,len(vocab),config)
        model.load_state_dict(checkpoint["state_dict"])
        for prompt_id,prompt in enumerate(meta["prompts"], 1):
            for temperature in config["temperatures"]:
                for sampling_seed in config["sampling_seeds"]:
                    tick = time.perf_counter()
                    words = generate(model,vocab,prompt,config["context_length"],temperature,
                                     sampling_seed,config["generation_words"])
                    samples.append({"model":kind,"training_seed":seed,"prompt_id":prompt_id,
                                    "prompt":prompt,"temperature":temperature,"sampling_seed":sampling_seed,
                                    "continuation":" ".join(words),"generation_seconds":time.perf_counter()-tick,
                                    **text_metrics(words)})
    write_json(out/"samples.json",samples)
    write_csv(out/"samples.csv",samples)
    sample_markdown = ["# All generated samples", "", "Unedited continuations; prompt excluded from diagnostics. Training seed 17 for every model.", ""]
    for s in samples:
        sample_markdown += [f"## {s['model']} | prompt {s['prompt_id']} | temperature {s['temperature']} | RNG {s['sampling_seed']}",
                            "",f"**Prompt:** {s['prompt']}","",s["continuation"],""]
    (out/"samples.md").write_text("\n".join(sample_markdown))
    # Smoothed train unigram baseline: a useful minimum performance reference.
    y = datasets["train"].tensors[1].numpy()
    probabilities = (np.bincount(y,minlength=len(vocab))+1)/(len(y)+len(vocab))
    test_y = datasets["test"].tensors[1].numpy()
    nll = -np.log(probabilities[test_y])
    baseline = {"name":"add-one train unigram", **__import__("src.metrics",fromlist=["summarize_predictions"]).summarize_predictions(
        nll, test_y==probabilities.argmax(), test_y)}
    write_json(out/"baseline.json", baseline)
    manifest.update(status="COMPLETE",finished_utc=datetime.now(timezone.utc).isoformat(),
                    total_wall_seconds=time.perf_counter()-start,best_model_by_validation=winner,
                    completed_runs=len(all_runs),generated_samples=len(samples))
    write_json(out/"run_manifest.json",manifest)
    freeze = subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=True)
    (out/"environment.txt").write_text(freeze.stdout)
    print(json.dumps({"status":"COMPLETE","winner_by_validation":winner,"seconds":manifest["total_wall_seconds"]}),flush=True)


if __name__ == "__main__":
    main()
