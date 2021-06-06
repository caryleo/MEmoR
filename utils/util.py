import json
from os import setgid
import pandas as pd
import numpy as np
import importlib
import pickle
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torch
from torch.nn.functional import embedding
from torch.nn.modules import module

from torch.utils.data import dataloader
from pymagnitude import Magnitude
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords

nltk_stopwords = stopwords.words('english')
# spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS # older version of spacy
stopwords = set(nltk_stopwords).union(spacy_stopwords)
porter = PorterStemmer()


def find_model_using_name(model_filename, model_name):

    modellib = importlib.import_module(model_filename)
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == model_name.lower():
            model = cls

    if not model:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, model_name))
        exit(0)

    return model

def create_model(config):
    model = find_model_using_name("model.model", config['model']['type'])
    instance = model(config)
    print("model [%s] was created" % (config['model']['type']))
    return instance

def create_dataloader(config):
    dataloader = find_model_using_name("data_loader.data_loaders", config['data_loader']['type'])
    instance = dataloader(config)
    print("dataset [%s] was created" % (config['data_loader']['type']))
    return instance

def create_trainer(model, criterion, metrics, logger, config, data_loader, valid_data_loader):
    trainer = find_model_using_name("trainer.trainer", config['trainer']['type'])
    instance = trainer(model, criterion, metrics,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                    )                  
    print("trainer [%s] was created" % (config['trainer']['type']))
    logger.info(model)
    return instance

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

##########################################
def to_pickle(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
##########################################

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)

##################################################################
def build_vocab(config, modal):
    # 针对给定的标签信息，汇总成词汇表，用于标注使用
    word2id = {'<unk>': 0}
    id2word = {0: '<unk>'}
    
    assert modal in ['visual', 'audio', 'text'], "Wrong Argument: MODAL"
    if modal == 'visual':
        faces_concepts = read_json(config['visual']['faces_concepts_file'])
        obj_concepts = read_json(config['visual']['obj_concepts_file'])
        action_concepts = read_json(config['visual']['action_concepts_file'])

        concepts = dict()
        for key in faces_concepts.keys():
            concepts[key] = faces_concepts[key] + obj_concepts[key] + action_concepts[key]
    elif modal == 'audio':
        concepts = read_json(config["audio"]['concepts_file'])
    else:
        concepts = read_json(config["text"]['concepts_file'])

    words = set()
    for _, items in concepts.items():
        words.union(items)

    words = list(words)

    for index, word in enumerate(words, start=1):
        word2id[word] = index
        id2word[index] = word

    return word2id, id2word

def build_kb(concept2id, config):
    # 原始实现，加载处理好的conceptNet还有NRC（或者SenticNet）根据标签空间构造矩阵
    # Calculate edge matrix
    conceptnet = load_pickle(config["knowledge"]["conceptnet_file"])
    filtered_conceptnet = filter_conceptnet(conceptnet, concept2id)
    filtered_conceptnet = remove_KB_duplicates(filtered_conceptnet)
    vocab_size = len(concept2id)
    edge_matrix = np.zeros((vocab_size, vocab_size))
    for k in filtered_conceptnet:
        for c,w in filtered_conceptnet[k]:
            edge_matrix[concept2id[k], concept2id[c]] = w

    kb_percentage = config["knowledge"]["kb_percentage"]
    if kb_percentage > 0: # 参考原始实现，给定一定的随机采样比例
            print("Keeping {0}% KB concepts...".format(kb_percentage * 100))
            edge_matrix = edge_matrix * (np.random.random((vocab_size,vocab_size)) < kb_percentage).astype(float)
    edge_matrix = torch.FloatTensor(edge_matrix)
    edge_matrix[torch.arange(vocab_size), torch.arange(vocab_size)] = 1

    # incorporate NRC VAD intensity
    NRC = load_pickle(config["knowledge"]["affectiveness_file"])
    affectiveness = np.zeros(vocab_size)
    for w, id in concept2id.items():
        VAD = get_emotion_intensity(NRC, w)
        affectiveness[id] = VAD
    affectiveness = torch.FloatTensor(affectiveness)

    return edge_matrix, affectiveness


def convert_examples_to_ids(concepts_list, concept2id):
    concepts_ids_list = []
    for concepts in concepts_list:
        concepts_ids_list.append([concept2id[concept] for concept in concepts])
    return concepts_ids_list



# conceptnet
def get_concept_embedding(concept2id, config):
    vectors = Magnitude(config["knowledge"]["embedding_file"])
    # 这里的实现和原始实现稍许不一样，如果对应不上会直接炸
    pretrained_word_embedding = np.zeros((len(concept2id), config["knowledge"]["embedding_dim"]))
    for word, index in concept2id.items():
        # 对于包含下划线的短语进行处理：二者加和
        if '_' in word:
            words = word.split('_')
            vector_phrase = np.zeros((1, config["knowledge"]["embedding_dim"]))
            for word_item in words:
                vector_phrase += vectors.query(word_item)
            pretrained_word_embedding[index] = vector_phrase
        else:
            pretrained_word_embedding[index] = vectors.query(word)

        assert pretrained_word_embedding[index].shape[1] == config["knowledge"]["embedding_dim"]
    return pretrained_word_embedding


def get_emotion_intensity(NRC, word):
    # 有啥复制啥，并且计算好，否则直接返回0.5，对应的SenticNet修改点位
    if word not in NRC:
        word = porter.stem(word)
        if word not in NRC:
            return 0.5
    v, a, d = NRC[word]
    a = a / 2
    return (np.linalg.norm(np.array([v, a]) - np.array([0.5, 0])) - 0.06467)/0.607468


# edge matrix construction
def filter_conceptnet(conceptnet, concept2id):
    # 整理conceptnet，将终止词过滤掉
    filtered_conceptnet = {}
    for k in conceptnet:
        if k in concept2id and k not in stopwords:
            filtered_conceptnet[k] = set()
            for c,w in conceptnet[k]:
                if c in concept2id and c not in stopwords and w>=1:
                    filtered_conceptnet[k].add((c,w))
    return filtered_conceptnet


# remove cases where the same concept has multiple weights
def remove_KB_duplicates(conceptnet):
    # 对于多个权重的情况，取最大权重
    filtered_conceptnet = {}
    for k in conceptnet:
        filtered_conceptnet[k] = set()
        concepts = set()
        filtered_concepts = sorted(conceptnet[k], key=lambda x: x[1], reverse=True)
        for c,w in filtered_concepts:
            if c not in concepts:
                filtered_conceptnet[k].add((c, w))
                concepts.add(c)
    return filtered_conceptnet

#####################################################