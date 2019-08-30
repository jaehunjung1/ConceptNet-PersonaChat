import torch
from torchtext import data
import ipdb
import json
from tqdm import tqdm
from openie.open_ie_api import *
from nltk import word_tokenize

# Torch device
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

        iterator = list(data.Iterator(self.dataset, len(self.dataset), device=device, repeat=False,
                                  sort_key=lambda x: -self.ENT1.vocab.stoi[x], train=False))[0]

        self.rel, self.ent1, self.ent2 = iterator.rel.unsqueeze(1).to(device), iterator.ent1.unsqueeze(1).to(device), \
                                         iterator.ent2.unsqueeze(1).to(device)


class Persona:
    def __init__(self, dir, concepetNet):
        self.path = dir
        self.conceptNet = conceptNet
        with open(self.path, 'r') as f:
            self.dataset = json.load(f)
        self.dialog = []  # list of chunked q-a lists, each word_tokenized
        self.q, self.a = None, None
        self.process_json()

    def process_json(self):
        for i, row in tqdm(enumerate(self.dataset)):
            # split multi turn dialogs into list of q-a pairs
            dialog = row['dialog']
            self.dataset[i]['dialog'] = [(dialog[i]['text'].lower(), dialog[i + 1]['text'].lower())
                                         for i in range(len(dialog) - 1)]
            self.dialog.extend([list((map(word_tokenize, self.dataset[i]['dialog'][j])))
                                for j in range(len(dialog)-1)])

            # remove needless k-v pairs
            self.dataset[i].pop('start_time')
            self.dataset[i].pop('end_time')
            self.dataset[i].pop('eval_score')
            self.dataset[i].pop('profile_match')
            self.dataset[i].pop('participant1_id')
            self.dataset[i].pop('participant2_id')

            # 'user_profile', 'bot_profile': persona
            # self.dataset[i]['user_profile'] = call_api_many(row['user_profile'], pagination_param=10000)

        field = data.Field(sequential=True, batch_first=True)
        field.vocab = conceptNet.ENT1.vocab

        q, a = tuple(zip(*self.dialog))
        self.q = field.numericalize(field.pad(q)).to(device)
        self.a = field.numericalize(field.pad(a)).to(device)


if __name__ == "__main__":
    print("Device:", device)

    conceptNet = ConceptNet("./conceptnet_data/train100k.tsv")
    persona = Persona("./persona_data/data_tolokers.json", conceptNet)

    qa_match = 0  # number of (entity1, entity2) in q-a pair
    for i in tqdm(range(persona.q.size(0))):

        row_q = persona.q[i]
        row_a = persona.a[i]

        ent_q = torch.unique(row_q)  # unique entities in query
        ent_a = torch.unique(row_a)  # unique entities in answer
        ent_q = ent_q.repeat(conceptNet.ent1.size(0), 1)
        ent_a = ent_a.repeat(conceptNet.ent2.size(0), 1)

        qa_match += sum([torch.sum((conceptNet.ent1 == ent_q) * (conceptNet.ent2 == ent_a)[:, i].unsqueeze(1))
                         for i in range(ent_a.size(1))])

    qa_avg = qa_match / persona.q.size(0)
    ipdb.set_trace()
