import torch
from torchtext import data
import ipdb
import json
from tqdm import tqdm
from openie.open_ie_api import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ConceptNet:
    def __init__(self, dir):
        self.path = dir
        self.REL = data.Field(sequential=False, batch_first=True)
        self.ENT1 = data.Field(sequential=False, batch_first=True)
        self.ENT2 = data.Field(sequential=False, batch_first=True)
        self.fields = [("rel", self.REL), ("ent1", self.ENT1), ("ent2", self.ENT2)]
        self.dataset = data.TabularDataset(path=self.path, format='tsv', fields=self.fields)

        self.rel, self.ent1, self.ent2 = None, None, None
        self.preprocess()

    def preprocess(self):
        self.REL.build_vocab(self.dataset, min_freq=1)  # 35 (including unk)
        self.ENT1.build_vocab(self.dataset.ent1, self.dataset.ent2, min_freq=1)  # 78095 (including unk)
        self.ENT2.vocab = self.ENT1.vocab

        # TODO add self.entities as set
        # self.ent1.build_vocab(self.dataset, min_freq=1)
        # self.ent2.build_vocab(self.dataset, min_freq=1)

        iter = list(data.Iterator(self.dataset, len(self.dataset), device=device, repeat=False,
                                  sort_key=lambda x: len(x.ent1), train=False))[0]

        self.rel, self.ent1, self.ent2 = iter.rel, iter.ent1, iter.ent2


class Persona:
    def __init__(self, dir):
        self.path = dir
        with open(self.path, 'r') as f:
            self.dataset = json.load(f)
        self.process_json()

    def process_json(self):
        for i, row in tqdm(enumerate(self.dataset)):
            # split multi turn dialogs into list of q-a pairs
            dialog = row['dialog']
            self.dataset[i]['dialog'] = [(dialog[i]['text'], dialog[i + 1]['text']) for i in range(len(dialog) - 1)]

            # # remove needless k-v pairs
            self.dataset[i].pop('start_time')
            self.dataset[i].pop('end_time')
            self.dataset[i].pop('eval_score')
            self.dataset[i].pop('profile_match')
            self.dataset[i].pop('participant1_id')
            self.dataset[i].pop('participant2_id')

            # 'user_profile', 'bot_profile': persona
            self.dataset[i]['user_profile'] = call_api_many(row['user_profile'], pagination_param=10000)

    # def extract_entity(self):


if __name__ == "__main__":
    conceptNet = ConceptNet("./conceptnet_data/train100k.tsv")
    # persona = Persona("./persona_data/data_tolokers.json")
    ipdb.set_trace()
