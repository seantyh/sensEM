from gensim.models import KeyedVectors
from CwnGraph import CwnImage, CwnSense
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class SenseData:
    sense_ids: np.ndarray
    sense_labels: List[str]
    sense_freqs: np.ndarray    
    sense_vecs: np.ndarray

class SenseKeyedVectors(KeyedVectors):
    def __init__(self, vector_size):
        super().__init__(vector_size)
        self.cwn = CwnImage.load("v.2020.05")

    @classmethod
    def load_from_kv(cls, fpath):
        kv = KeyedVectors.load_word2vec_format(fpath, binary=True)
        skv = SenseKeyedVectors(kv.vector_size)
        skv.__dict__.update(kv.__dict__)
        return skv

    def query_sense(self, term):
        if "-" not in term:
            return term
        cwn_id = term[term.index("-")+1:]
        try:
            sense = CwnSense(cwn_id, self.cwn)
            return sense.head_word + ": " + sense.definition
        except Exception:
            return term

    def get_token_idx(self, sense: CwnSense):
        tok = f"{sense.head_word}-{sense.id}"
        tok_idx = self.key_to_index.get(tok, -1)
        return tok_idx

    def query_sense_freq(self, sense: CwnSense):
        tok_idx = self.get_token_idx(sense)
        if tok_idx < 0:
            return 0
        else:
            return self.expandos["count"][tok_idx]

    def query_vector(self, sense: CwnSense):
        tok_idx = self.get_token_idx(sense)
        if tok_idx < 0:
            return None
        else:
            return self.get_vector(tok_idx, norm=True)
    
    def show_neighbors_bysenses(self, lemma):
        senses = self.cwn.find_all_senses(lemma)
        ret = {}  
        for sense_x in senses:
            key = f"{lemma}-{sense_x.id}"
            sense_freq = self.query_sense_freq(sense_x)
            if key not in self:
                label = f"{lemma}({sense_x.id}) f={sense_freq: 5d}"
                ret[label] = []
            else:
                neighs = self.most_similar(key)            
                label = f"{lemma}({sense_x.id}) f={sense_freq: 5d}: {sense_x.definition})"
                ret[label] = [(x[:x.index("-")], v) for x, v in neighs]
        return ret

    def make_sense_vectors(self, lemma):
        senses = self.cwn.find_all_senses(lemma)
        sense_ids = []
        sense_labels = []
        sense_freqs = []
        vecs = []

        for sense_x in senses:
            vec_x = self.query_vector(sense_x)
            if vec_x is None: continue
            vecs.append(vec_x)
            sense_ids.append(sense_x.id)
            try:
                ex0 = sense_x.all_examples()[0]
                ex = ex0[ex0.index("<")-3: ex0.index(">")+4]
            except:
                ex = ""
            sense_labels.append(f"[{sense_x.id}]{sense_x.definition}: {ex}")
            sense_freqs.append(self.query_sense_freq(sense_x))

        if len(vecs):
            stack_vecs = np.vstack(vecs)
        else:
            stack_vecs = np.array([])

        return SenseData(sense_ids, sense_labels, sense_freqs, 
                stack_vecs)


