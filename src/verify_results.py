"""Recompute checkpoint test metrics and generation; audit evidence completeness."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import prepare
from .models import from_config
from .metrics import evaluate, generate
from .run import write_json
from .analyze import read_csv


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",default="results")
    a=p.parse_args()
    root=Path(a.results)
    manifest=json.loads((root/"run_manifest.json").read_text())
    assert manifest["status"]=="COMPLETE"
    cfg=manifest["config"]
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    for path,checksum in manifest["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==checksum,path
    ds,vocab,_,_=prepare("data/raw/alice_pg11.txt",cfg)
    rows=read_csv(root/"runs.csv")
    assert len(rows)==len(cfg["models"])*len(cfg["seeds"])
    history=read_csv(root/"history.csv")
    assert len(history)==len(rows)*cfg["epochs"]
    model_cache={}
    for r in rows:
        kind,seed=r["model"],int(r["seed"])
        ckpt=torch.load(root/"checkpoints"/f"{kind}_seed{seed}.pt",weights_only=True,map_location="cpu")
        model=from_config(kind,len(vocab),cfg)
        model.load_state_dict(ckpt["state_dict"])
        metrics,arrays=evaluate(model,DataLoader(ds["test"],batch_size=cfg["batch_size"],shuffle=False))
        for key in ["loss","perplexity","accuracy","known_word_accuracy"]:
            assert np.isclose(metrics[key],float(r[f"test_{key}"]),rtol=1e-7,atol=1e-9),(kind,seed,key)
        saved=np.load(root/"predictions"/f"{kind}_seed{seed}_test.npz")
        for key in arrays:
            np.testing.assert_array_equal(arrays[key],saved[key])
        run_history=[h for h in history if h["model"]==kind and int(h["seed"])==seed]
        best=min(run_history,key=lambda h:float(h["validation_loss"]))
        assert int(best["epoch"])==ckpt["epoch"]==int(r["best_epoch"])
        assert int(r["epochs_trained"])==cfg["epochs"]
        if seed==cfg["seeds"][0]:
            model_cache[kind]=model
    samples=json.loads((root/"samples.json").read_text())
    expected=len(cfg["models"])*2*len(cfg["temperatures"])*len(cfg["sampling_seeds"])
    assert len(samples)==expected
    # One continuation per model checks saved weights/vocabulary/sampling end to end.
    for kind,model in model_cache.items():
        s=next(s for s in samples if s["model"]==kind)
        words=generate(model,vocab,s["prompt"],cfg["context_length"],s["temperature"],s["sampling_seed"],cfg["generation_words"])
        assert " ".join(words)==s["continuation"]
    for s in samples:
        assert len(s["continuation"].split())==cfg["generation_words"]
    report={"status":"PASS","checkpoint_test_metrics_recomputed":len(rows),
            "test_prediction_arrays_exactly_matched":len(rows),"selected_epochs_verified":len(rows),
            "training_history_rows":len(history),"samples_count_and_length_verified":len(samples),
            "saved_generations_exactly_reproduced":len(model_cache),"training_source_hashes_verified":len(manifest["source_sha256"])}
    write_json(root/"verification.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
