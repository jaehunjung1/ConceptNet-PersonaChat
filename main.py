import torch
from torchtext import data
import ipdb
import json
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake
import string
import jsonlines


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
    def __init__(self, dir, conceptNet):
        self.path = dir
        self.conceptNet = conceptNet
        with open(self.path, 'r') as f:
            self.dataset = json.load(f)
        self.dialog = []  # [{"id": 0, "speakerA": [("i", "N") ...], "speakerB": ~}, ...]
        self.candidates = [] # [[id, [lemmatized speakerA], [lemmatized speakerB]], ...]
        self.q, self.a = None, None

        self.lemmatizer = WordNetLemmatizer()
        self.tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}

    def _proc_with_stopwords(self, obj):
        return [x for x in obj if x[0] not in self.stops]

    def _lemmatize(self, pos):
        lem = []
        for w, tag in pos:
            tag_proj = self.tag_dict.get(tag[0], None)
            if tag_proj is not None:
                lem.append(self.lemmatizer.lemmatize(w, tag_proj))
        return lem

    def process(self, filt):
        if filt == 'stopwords':
            self.stops = set(stopwords.words('english'))
        elif filt == 'rake':
            self.rake = Rake(max_length=1, punctuations=string.punctuation + "’")
            topk = 5

        for i, row in tqdm(enumerate(self.dataset)):
            # split multi turn dialogs into list of q-a pairs
            dialog = row['dialog']
            self.dataset[i]['dialog'] = [(dialog[idx]['text'].lower(), dialog[idx + 1]['text'].lower())
                                         for idx in range(len(dialog) - 1)]
            self.dialog.append({"id": len(self.dialog),
                                "speakerA": self.dataset[i]['dialog'][0],
                                "speakerB": self.dataset[i]['dialog'][1]})

            for j in range(len(dialog) - 1):
                q_pos = nltk.pos_tag(nltk.word_tokenize(self.dataset[i]['dialog'][j][0]))
                a_pos = nltk.pos_tag(nltk.word_tokenize(self.dataset[i]['dialog'][j][1]))

                if filt == 'stopwords':  # stopwords option: exclude all stopwords from speakerA, speakerB
                    q_pos = self._proc_with_stopwords(q_pos)
                    a_pos = self._proc_with_stopwords(a_pos)
                elif filt == 'rake':  # rake option: extract entities using rake algorithm
                    self.rake.extract_keywords_from_text(self.dataset[i]['dialog'][j][0])
                    raked = self.rake.get_ranked_phrases()[:topk]
                    q_pos = list(set([x for x in q_pos if x[0] in raked]))

                    self.rake.extract_keywords_from_text(self.dataset[i]['dialog'][j][1])
                    raked = self.rake.get_ranked_phrases()[:topk]
                    a_pos = list(set([x for x in a_pos if x[0] in raked]))

                if len(q_pos) > 0 and len(a_pos) > 0:
                    q_lem = self._lemmatize(q_pos)
                    a_lem = self._lemmatize(a_pos)
                    self.candidates.append([len(self.dialog)-1+j, q_lem, a_lem])

            # 'user_profile', 'bot_profile': persona

        self.field = data.Field(sequential=True, batch_first=True)
        self.field.vocab = self.conceptNet.ENT1.vocab

        self.id, q, a = tuple(zip(*self.candidates))
        self.q = self.field.numericalize(self.field.pad(q)).to(device)
        self.a = self.field.numericalize(self.field.pad(a)).to(device)


if __name__ == "__main__":
    # Configuration
    filt = 'stopwords'
    conceptNet_dir = "./conceptnet_data/train100k.tsv"
    persona_dir = "./persona_data/train_both_original_no_cands.json"
    save_dir = "./output/result.json"

    # Torch device
    gpu_idx = input("GPU IDX: ")
    device = torch.device('cuda:' + gpu_idx if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Fix random seed
    torch.manual_seed(0)

    conceptNet = ConceptNet(conceptNet_dir)
    persona = Persona(persona_dir, conceptNet)
    persona.process(filt)

    writer = jsonlines.open(save_dir, 'w')

    qa_match_all = 0.0  # number of (entity1, entity2) in q-a pair
    for i in tqdm(range(persona.q.size(0))):
        row_id = persona.id[i]
        row_q = persona.q[i]
        row_a = persona.a[i]

        ent_q = row_q.repeat(conceptNet.ent1.size(0), 1)
        ent_a = row_a.repeat(conceptNet.ent2.size(0), 1)
        qa_match_pos = [((conceptNet.ent1 == ent_q)[:, i].unsqueeze(1) * (conceptNet.ent2 == ent_a)).nonzero() for i in
                        range(ent_q.size(1))]  # 0
        has_qa_match = torch.tensor([t.nelement() > 0 for t in qa_match_pos])
        qa_match_q_idx = has_qa_match.nonzero().squeeze(-1)

        for c_q_pos in qa_match_q_idx:
            r_pos = qa_match_pos[c_q_pos][0][0].item()
            c_a_pos = qa_match_pos[c_q_pos][0][1].item()
            match_tok_q = persona.field.vocab.itos[ent_q[r_pos, c_q_pos].item()]
            match_tok_a = persona.field.vocab.itos[ent_a[r_pos, c_a_pos].item()]
            d = {'id': row_id, 'tok_q': match_tok_q, 'tok_a': match_tok_a}
            writer.write(d)

        qa_match_all += sum([t.nelement() for t in qa_match_pos])

    qa_avg = qa_match_all / persona.q.size(0)
    print('{} : {}'.format(filt, qa_avg))
