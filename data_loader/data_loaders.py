import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import torch
from base import BaseDataLoader
from utils import read_json, build_vocab, get_concept_embedding, build_kb, convert_examples_to_ids
import numpy as np
import pandas as pd
from tqdm import tqdm
from features import AudioFeatureExtractor, TextFeatureExtractor, VisualFeatureExtractor, PersonalityFeatureExtractor
from pymagnitude import Magnitude

EMOTIONS = ["neutral","joy","anger","disgust","sadness","surprise","fear","anticipation","trust","serenity","interest","annoyance","boredom","distraction"]

class MEmoRDataset(data.Dataset):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        annos = read_json(config['anno_file'])[config['emo_type']]

        ##################################################
        # 作者将这部分代码注释掉了，就是用是否只用训练数据集进行训练
        # 在train_test模式下，所有的数据都应该预先处理
        ids = []
        tmp_annos = []
        with open(config['id_file']) as fin:
            for line in fin.readlines():
                ids.append(int(line.strip()))
        
        for jj, anno in enumerate(annos):
            if jj in ids:
                tmp_annos.append(anno)
        annos = tmp_annos
        ##################################################
            
        emo_num = 9 if config['emo_type'] == 'primary' else 14
        self.emotion_classes = EMOTIONS[:emo_num]
        
        data = read_json(config['data_file'])
        self.visual_features, self.audio_features, self.text_features = [], [], []
        self.visual_valids, self.audio_valids, self.text_valids = [], [], []

        ################################
        # 用来保存概念，用来后面准备加知识图
        self.visual_concepts, self.audio_concepts, self.text_concepts = list(), list(), list()
        self.visual_concepts_lengths, self.audio_concepts_lengths, self.text_concepts_lengths = list(), list(), list()
        ################################

        self.labels = []
        self.charcaters_seq = []
        self.time_seq = []
        self.target_loc = []
        self.seg_len = [] 
        self.n_character = []
        vfe = VisualFeatureExtractor(config)
        afe = AudioFeatureExtractor(config)
        tfe = TextFeatureExtractor(config)
        pfe = PersonalityFeatureExtractor(config)
        self.personality_list = pfe.get_features() # n_c
        self.personality_features = []

        ###################################
        print("Processing Concepts")
        self.concept2id_v, self.id2concept_v = build_vocab(config, 'visual')
        self.concept2id_a, self.id2concept_a = build_vocab(config, 'audio')
        self.concept2id_t, self.id2concept_t = build_vocab(config, 'text')

        vfe.concepts2id = self.concept2id_v
        afe.concepts2id = self.concept2id_a
        tfe.concepts2id = self.concept2id_t

        # print(self.concept2id_t)

        # print(afe.concepts2id)

        config["visual"]["concept_size"] = len(self.concept2id_v)
        config["audio"]["concept_size"] = len(self.concept2id_a)
        config["text"]["concept_size"] = len(self.concept2id_t)
        ###################################

        ###################################
        print("Processing Knowledge") 
        vectors = Magnitude(config["knowledge"]["embedding_file"])
        self.embedding_concept_v = get_concept_embedding(self.concept2id_v, config, vectors)
        self.embedding_concept_a = get_concept_embedding(self.concept2id_a, config, vectors)
        self.embedding_concept_t = get_concept_embedding(self.concept2id_t, config, vectors)

        self.edge_matrix_v, self.affectiveness_v = build_kb(self.concept2id_v, config, "visual")
        self.edge_matrix_a, self.affectiveness_a = build_kb(self.concept2id_a, config, "audio")
        self.edge_matrix_t, self.affectiveness_t = build_kb(self.concept2id_t, config, "text")
        ###################################
        
        print('Processing Samples...')
        for jj, anno in enumerate(tqdm(annos)):
            # if jj >= 300: break
            clip = anno['clip']
            target_character = anno['character']
            target_moment = anno['moment']
            on_characters = data[clip]['on_character']
            if target_character not in on_characters:
                on_characters.append(target_character)
            on_characters = sorted(on_characters)
            
            charcaters_seq, time_seq, target_loc, personality_seq = [], [], [], []
            

            for ii in range(len(data[clip]['seg_start'])):
                for character in on_characters:
                    charcaters_seq.append([0 if character != i else 1 for i in range(len(config['speakers']))])
                    time_seq.append(ii)
                    personality_seq.append(self.personality_list[character])
                    if character == target_character and data[clip]['seg_start'][ii] <= target_moment < data[clip]['seg_end'][ii]:
                        target_loc.append(1)
                    else:
                        target_loc.append(0)
            # for character in on_characters:
            #     for ii in range(len(data[clip]['seg_start'])):
            #         charcaters_seq.append([0 if character != i else 1 for i in range(len(config['speakers']))])
            #         time_seq.append(ii)
            #         personality_seq.append(self.personality_list[character])
            #         if character == target_character and data[clip]['seg_start'][ii] <= target_moment < data[clip]['seg_end'][ii]:
            #             target_loc.append(1)
            #         else:
            #             target_loc.append(0)
            
            ####################################################
            # 什么c就是对应的概念，读到列表里面，暂时没想好动作特征咋处理
            vf, v_valid, vc = vfe.get_feature(anno['clip'], target_character) # seqlen * n_c, dim_features_v
            af, a_valid, ac = afe.get_feature(anno['clip'], target_character) # seqlen * n_c, dim_features_a
            tf, t_valid, tc = tfe.get_feature(anno['clip'], target_character) # seqlen * n_c, dim_features_t
            ####################################################
            
            self.n_character.append(len(on_characters))
            self.seg_len.append(len(data[clip]['seg_start']))
    
            self.personality_features.append(torch.stack(personality_seq)) # num_anno, seqlen * n_c, dim_features_p
            self.charcaters_seq.append(torch.tensor(charcaters_seq)) # num_anno, seqlen * n_c, some
            self.time_seq.append(torch.tensor(time_seq)) # num_anno, seqlen * n_c, some
            self.target_loc.append(torch.tensor(target_loc, dtype=torch.int8)) # num_anno, seqlen * n_c
            self.visual_features.append(vf) # num_anno, seqlen * n_c, dim_features_v
            self.audio_features.append(af) # num_anno, seqlen * n_c, dim_features_a
            self.text_features.append(tf) # num_anno, seqlen * n_c, dim_features_t
            self.visual_valids.append(v_valid) # num_anno, seqlen * n_c
            self.audio_valids.append(a_valid) # num_anno, seqlen * n_c
            self.text_valids.append(t_valid) # num_anno, seqlen * n_c

            #######################################################
            # 对应的保存，按照样本对应
            lengths = list()
            vc_new = list()
            for concepts in vc:
                new = torch.zeros(512, dtype=torch.long)
                lengths.append(concepts.size(0))
                new[:concepts.size(0)] = concepts[:]
                vc_new.append(new)
            self.visual_concepts_lengths.append(torch.tensor(lengths, dtype=torch.int8)) # num_anno, seqlen

            # assert len(vc_new) == len(vc) and len(vc_new) == len(data[clip]['seg_start'])

            ac_new = list()
            lengths = list()
            for concepts in ac:
                # print(concepts)
                new = torch.zeros(512, dtype=torch.long) # max_num_concept
                lengths.append(concepts.size(0))
                new[:concepts.size(0)] = concepts[:]
                ac_new.append(new)
            self.audio_concepts_lengths.append(torch.tensor(lengths, dtype=torch.int8)) # num_anno, seqlen

            tc_new = list()
            lengths = list()
            for concepts in tc:
                new = torch.zeros(512, dtype=torch.long)
                lengths.append(concepts.size(0))
                new[:concepts.size(0)] = concepts[:]
                tc_new.append(new)
            self.text_concepts_lengths.append(torch.tensor(lengths, dtype=torch.int8)) # num_anno, seqlen

            self.visual_concepts.append(torch.stack(vc_new, dim=0)) # num_anno, seqlen, max_num_concept
            # assert torch.stack(vc_new, dim=0).size(0) == len(data[clip]['seg_start'])
            self.audio_concepts.append(torch.stack(ac_new, dim=0)) # num_anno, seqlen, max_num_concept
            self.text_concepts.append(torch.stack(tc_new, dim=0)) # num_anno, seqlen, max_num_concept
            #######################################################

            self.labels.append(self.emotion_classes.index(anno['emotion']))            
        

    def __getitem__(self, index):
        ########################################
        # 增加了对应的输出调整，按照样本对应
        # print("GETITEM", self.visual_concepts[index].size())
        return torch.tensor([self.labels[index]]), \
            self.visual_features[index], \
            self.audio_features[index], \
            self.text_features[index], \
            self.personality_features[index], \
            self.visual_valids[index], \
            self.audio_valids[index], \
            self.text_valids[index], \
            self.visual_concepts[index],\
            self.audio_concepts[index],\
            self.text_concepts[index],\
            self.visual_concepts_lengths[index],\
            self.audio_concepts_lengths[index],\
            self.text_concepts_lengths[index],\
            self.target_loc[index], \
            torch.tensor([1] * len(self.time_seq[index]), dtype=torch.int8), \
            torch.tensor([self.seg_len[index]], dtype=torch.int8), \
            torch.tensor([self.n_character[index]], dtype=torch.int8)
        #######################################
            

    def __len__(self):
        return len(self.visual_features)

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True) for i in dat]

    def statistics(self):
        all_emotion = [0] * len(self.emotion_classes)
        for emotion in self.labels:
            all_emotion[emotion] += 1
        return all_emotion


class MEmoRDataLoader(BaseDataLoader):
    def __init__(self, config, training=True):
        data_loader_config = config['data_loader']['args']
        self.seed = data_loader_config['seed']
        self.dataset = MEmoRDataset(config)
        self.emotion_nums = self.dataset.statistics()

        #####################################
        ## 修改：将验证集强制为指定文件
        self.val_file = False
        if config.val_file:
            self.val_file = True
            print('Caution! Loading', config['val_id_file'], 'as validation set')
            test_list = list()
            with open(config['val_id_file']) as val_file:
                for line in val_file.readlines():
                    test_list.append(int(line))
            self.valid_idx = np.array(test_list)
        #######################################

        super().__init__(self.dataset, data_loader_config['batch_size'], data_loader_config['shuffle'], data_loader_config['validation_split'], data_loader_config['num_workers'], collate_fn=self.dataset.collate_fn)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(self.seed)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        ##################
        # 制定了测试文件（valtest）就将其作为验证集，否则将训练集分拆出测试集
        if self.val_file:
            valid_idx = self.valid_idx
            train_idx = np.array([idx for idx in idx_full if idx not in valid_idx])
        else:
            valid_idx = idx_full[0:len_valid]
            train_idx = np.delete(idx_full, np.arange(0, len_valid))

        #######################
        weights_per_class = 1. / torch.tensor(self.emotion_nums, dtype=torch.float)
        weights = [0] * self.n_samples
        for idx in range(self.n_samples):
            if idx in valid_idx:
                weights[idx] = 0.
            else:
                label = self.dataset[idx][0]
                weights[idx] = weights_per_class[label]
        weights = torch.tensor(weights)
        train_sampler = data.WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    
        valid_sampler = data.SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler
