import math
import unittest
from pathlib import Path
import numpy as np
import torch
from src.data import prepare, tokenize
from src.models import from_config
from src.metrics import generate, summarize_predictions
import json


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads(Path("configs/experiment.json").read_text())
        cls.datasets,cls.vocab,cls.origins,cls.meta=prepare("data/raw/alice_pg11.txt",cls.config)

    def test_splits_have_disjoint_source_positions(self):
        source_sets=[]
        for origin in self.origins.values():
            source_sets.append({(int(ch),p) for ch,start,end in origin for p in range(int(start),int(end)+1)})
        for i in range(3):
            for j in range(i):
                self.assertFalse(source_sets[i] & source_sets[j])

    def test_inputs_end_before_targets_and_have_fixed_length(self):
        for split,dataset in self.datasets.items():
            x,y=dataset.tensors
            self.assertEqual(x.shape,(len(y),16))
            self.assertEqual(y.ndim,1)
            self.assertTrue(np.all(self.origins[split][:,2]-self.origins[split][:,1]==16))
            self.assertLess(int(x.max()),len(self.vocab))

    def test_vocabulary_is_train_only(self):
        from src.data import read_chapters, make_vocab
        chapters,_=read_chapters("data/raw/alice_pg11.txt")
        vocab,_=make_vocab(chapters,2)
        changed=dict(chapters)
        changed[12]=["futureleak"]*500
        self.assertEqual(vocab,make_vocab(changed,2)[0])
        self.assertNotIn("futureleak",vocab)

    def test_word_tokenization(self):
        self.assertEqual(tokenize("Alice’s CAT, 123 can't-stop."),["alice's","cat","can't","stop"])

    def test_all_encoders_predict_one_word_from_observed_window(self):
        for kind in self.config["models"]:
            model=from_config(kind,len(self.vocab),self.config).eval()
            x=self.datasets["train"].tensors[0][:2]
            logits=model(x)
            self.assertEqual(logits.shape,(2,len(self.vocab)))
            # Target values cannot affect inference: forward accepts only x.
            self.assertTrue(torch.equal(logits,model(x.clone())))

    def test_bidirectional_representation_uses_both_final_hidden_states(self):
        model=from_config("birnn",len(self.vocab),self.config).eval()
        x=self.datasets["train"].tensors[0][:2]
        _,h=model.recurrent(model.embedding(x))
        expected=model.output(torch.cat([h[-2],h[-1]],1))
        self.assertTrue(torch.equal(expected,model(x)))

    def test_perplexity_uses_token_weighted_mean_nll(self):
        m=summarize_predictions([1.,2.,3.],[True,False,True],[1,0,2])
        self.assertAlmostEqual(m["perplexity"],math.exp(2))
        self.assertAlmostEqual(m["accuracy"],2/3)
        self.assertEqual(m["known_word_accuracy"],1)

    def test_sampling_reproducible_and_rejects_invalid_input(self):
        model=from_config("rnn",len(self.vocab),self.config)
        args=(model,self.vocab,self.meta["prompts"][0],16,1.0,101,8)
        self.assertEqual(generate(*args),generate(*args))
        with self.assertRaises(ValueError):
            generate(model,self.vocab,"alice",16,0,1,8)
        with self.assertRaises(ValueError):
            generate(model,self.vocab,"",16,1,1,8)


if __name__=="__main__":
    unittest.main()
