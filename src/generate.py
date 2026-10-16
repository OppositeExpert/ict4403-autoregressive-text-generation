"""Generate from any saved checkpoint without retraining."""
import argparse
import json
from pathlib import Path
import torch
from .models import from_config
from .metrics import generate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="results/checkpoints/gru_seed17.pt")
    p.add_argument("--vocab",default="results/vocab.json")
    p.add_argument("--prompt",default="alice was beginning to get very tired of sitting by her sister on the bank")
    p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--seed",type=int,default=101)
    p.add_argument("--words",type=int,default=60)
    a=p.parse_args()
    torch.set_num_threads(4)
    c=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    v=json.loads(Path(a.vocab).read_text())
    m=from_config(c["model"],len(v),c["config"])
    m.load_state_dict(c["state_dict"])
    words=generate(m,v,a.prompt,c["config"]["context_length"],a.temperature,a.seed,a.words)
    print(a.prompt+"\n"+" ".join(words))


if __name__=="__main__":
    main()
