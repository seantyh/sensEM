## Adapted from 10.01-tag-asbc.ipynb

import sys
import os
if "../src" not in sys.path:
    sys.path.append("../src")
if "../../pyASBC/src" not in sys.path:
    sys.path.append("../../pyASBC/src")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
from pathlib import Path
from itertools import islice
from pyASBC import Asbc5Corpus
from dotted_wsd import DottedWsdTagger
from tqdm.auto import tqdm

corpus = Asbc5Corpus("../../pyASBC/data")
tagger = DottedWsdTagger()
n_sentence = 1_396_133

tok_func = lambda x: (x[0], x[1])
tagged_func = lambda x: (*x[:2], *parse_prediction(x[2]))
    
def parse_prediction(pred_str: str):
    fields = pred_str.split(" ")
    if len(fields) == 3:
        return fields[0][1:-1], fields[1], fields[2][1:-1]
    else:
        return ("", "", "")

if __name__ == "__main__":
    out_dir = Path("../data/dt-asbc")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    batch_size = 10_000
# batch_size = 20
batch_idx = 0
n_batch = math.ceil(n_sentence / batch_size)
path_templ = f"asbc_dotted_tagged_{{batch_idx:03d}}-of-{n_batch}.txt"
batch_path = out_dir / path_templ.format(batch_idx=batch_idx)
if not batch_path.exists():
    fout = batch_path.open("w", encoding="utf-8")
else:
    fout = None

for sent_i, sent_x in enumerate(tqdm(corpus.iter_sentences(), total=n_sentence)):
    ## if file already exists, fout is None, 
    ## then skip the tagging part
    if fout:
        ## tagging
        try:
            tok_seq = list(map(tok_func, sent_x))
            sense_tagged = tagger.sense_tag_per_sentence(tok_seq)
            sense_tagged = list(map(tagged_func, sense_tagged))
            for tok_i, tagged_tok in enumerate(sense_tagged):
                if tagged_tok[2] and not tagged_tok[2].startswith("RP:"):
                    # it is tagged
                    fout.write(f"{tagged_tok[0]}-{tagged_tok[2]}")
                else:
                    # it is not tagged
                    fout.write(f"{tagged_tok[0]}-{tagged_tok[1]}")
                if tok_i < len(sense_tagged)-1:
                    fout.write(" ")
            fout.write("\n")
        except Exception as ex:
            print(ex)
    
    if (sent_i+1) % batch_size == 0:
        if fout: fout.close()
        batch_idx += 1
        
        if batch_idx > 6:
            break                    
        batch_path = out_dir / path_templ.format(batch_idx=batch_idx)
        if not batch_path.exists():
            fout = Path(batch_path).open("w", encoding="utf-8")
        else:
            fout = None

    if fout:
        fout.close()