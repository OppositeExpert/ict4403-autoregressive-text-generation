"""Regenerate publication plots and summary tables from saved evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .run import write_csv

LABELS={"rnn":"Simple RNN","birnn":"Bidirectional RNN","lstm":"LSTM","gru":"GRU"}
COLORS=["#276DA8","#D78025","#309171","#9D579A"]


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",default="results")
    a=p.parse_args()
    root=Path(a.results)
    plots=root/"plots"
    plots.mkdir(exist_ok=True)
    summary=read_csv(root/"summary.csv")
    history=read_csv(root/"history.csv")
    samples=json.loads((root/"samples.json").read_text())
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "figure.dpi":140,"savefig.dpi":180})
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True,layout="constrained")
    for ax,kind,color in zip(axes.flat,LABELS,COLORS):
        rec=[r for r in history if r["model"]==kind]
        epochs=sorted({int(r["epoch"]) for r in rec})
        for metric,style,label in [("train_online_loss","--","Training online"),("validation_loss","-","Validation")]:
            values=np.array([[float(r[metric]) for r in rec if int(r["epoch"])==epoch] for epoch in epochs])
            mean,sd=values.mean(1),values.std(1,ddof=1)
            ax.plot(epochs,mean,style,color=color,label=label)
            ax.fill_between(epochs,mean-sd,mean+sd,color=color,alpha=.13)
        ax.set_title(LABELS[kind]);ax.set_xlabel("Epoch");ax.set_ylabel("Cross entropy (nats)")
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle("Fixed 18-epoch budget | mean and SD across 3 training seeds")
    fig.savefig(plots/"learning_curves.png");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout="constrained")
    for ax,key,title in zip(axes,["test_perplexity","test_accuracy","training_seconds"],
                           ["Test perplexity (lower is better)","Test top-1 accuracy (%)","Training wall time (s)"]):
        scale=100 if key=="test_accuracy" else 1
        means=[float(r[f"{key}_mean"])*scale for r in summary]
        sds=[float(r[f"{key}_sd"])*scale for r in summary]
        ax.bar(range(4),means,yerr=sds,color=COLORS,capsize=4)
        ax.set_xticks(range(4),["RNN","BiRNN","LSTM","GRU"])
        ax.set_title(title,fontsize=10);ax.grid(axis="y",alpha=.15)
        for i,value in enumerate(means):
            ax.annotate(f"{value:.2f}",(i,value),xytext=(0,9),textcoords="offset points",ha="center",fontsize=8)
        ax.set_ylim(0,max(np.array(means)+np.array(sds))*1.23)
    fig.suptitle("Held-out chapters 11-12 | mean ± SD over 3 training seeds")
    fig.savefig(plots/"model_comparison.png");plt.close(fig)
    temperature_summary=[]
    metric_keys=["distinct_1","distinct_2","repeated_bigram_fraction","word_entropy_bits","unk_fraction"]
    for kind in LABELS:
        for temp in [0.5,1.,1.5]:
            group=[s for s in samples if s["model"]==kind and s["temperature"]==temp]
            record={"model":kind,"temperature":temp,"samples":len(group)}
            for key in metric_keys:
                values=np.array([s[key] for s in group])
                record[key+"_mean"]=float(values.mean())
                record[key+"_sd"]=float(values.std(ddof=1))
            temperature_summary.append(record)
    write_csv(root/"temperature_summary.csv",temperature_summary)
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout="constrained")
    for ax,key,title in zip(axes,["distinct_1","repeated_bigram_fraction","unk_fraction"],
                           ["Distinct words / generated words","Repeated bigram fraction","Unknown word fraction"]):
        for kind,color in zip(LABELS,COLORS):
            rows=[r for r in temperature_summary if r["model"]==kind]
            ax.errorbar([r["temperature"] for r in rows],[r[key+"_mean"] for r in rows],
                        yerr=[r[key+"_sd"] for r in rows],label=LABELS[kind],color=color,marker="o",capsize=3)
        ax.set_title(title,fontsize=10);ax.set_xlabel("Temperature");ax.set_ylim(-.02,1.02)
        ax.set_xticks([.5,1,1.5]);ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle("Generation diagnostics | mean ± SD across 2 prompts × 3 sampling seeds")
    fig.savefig(plots/"temperature_effects.png");plt.close(fig)
    lines=["| Model | Parameters | Validation loss | Test loss | Test perplexity | Test accuracy | Train seconds |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        def val(key,scale=1,digits=3):
            return f"{float(r[key+'_mean'])*scale:.{digits}f} ± {float(r[key+'_sd'])*scale:.{digits}f}"
        lines.append(f"| {LABELS[r['model']]} | {int(r['parameters']):,} | {val('validation_loss')} | {val('test_loss')} | {val('test_perplexity',digits=2)} | {val('test_accuracy',100,2)}% | {val('training_seconds',digits=2)} |")
    (root/"comparison.md").write_text("\n".join(lines)+"\n")
    print("Wrote 3 plots, temperature_summary.csv and comparison.md")


if __name__=="__main__":
    main()
