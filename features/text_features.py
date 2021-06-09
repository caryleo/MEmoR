import torch
from utils import read_json, convert_examples_to_ids
from base import BaseFeatureExtractor


class TextFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__()
        print("Initializing TextFeatureExtractor...")
        self.feature_file = config["text"]["feature_file"]
        self.feature_dim = config["text"]["feature_dim"]
        self.features = read_json(self.feature_file)
        self.data = read_json(config["data_file"])
        ########
        self.concepts = read_json(config["text"]['concepts_file'])
        self.concepts2id = None
        ########
        self.missing_tensor = torch.zeros((self.feature_dim))

    def get_feature(self, clip, target_character):
        on_characters = self.data[clip]["on_character"]
        if target_character not in on_characters:
            on_characters.append(target_character)
        on_characters = sorted(on_characters)
        speakers = self.data[clip]["speakers"]
        seg_ori_ind = self.data[clip]["seg_ori_ind"]
        ##############
        ret_concepts = list()
        ##############
        ret = []
        ret_valid = []
        
        for ii, speaker in enumerate(speakers):
            index = "{}+{}".format(clip, ii) # 妈的原作者实现有问题
            index_concept = "{}+{}".format(clip, seg_ori_ind[ii])
            for character in on_characters:
                if character == speaker:
                    ret.append(torch.tensor(self.features[index]))
                    ret_valid.append(1)
                else:
                    ret.append(self.missing_tensor)
                    ret_valid.append(0)

            ret_concepts.append(torch.LongTensor(convert_examples_to_ids(self.concepts[index_concept] if index_concept in self.concepts else [], self.concepts2id)) ) # seqlen

        # for character in on_characters:
        #     for ii, speaker in enumerate(speakers):
        #         if character == speaker:
        #             index = "{}+{}".format(clip, ii)
        #             ret.append(torch.tensor(self.features[index]))
        #             ret_valid.append(1)
        #             ###############
        #             ret_concepts.append([self.concept2id[concept] for concept in self.concepts[index]])
        #             ###############
        #         else:
        #             ret.append(self.missing_tensor)
        #             ret_valid.append(0)
        #             ###############
        #             # 缺省状态，对应的就是空的标签列表
        #             ret_concepts.append([])
        #             ##############

        ret = torch.stack(ret, dim=0) # seqlen * n_c, dim_feature_t
        ret_valid = torch.tensor(ret_valid, dtype=torch.int8) # seqlen * n_c
        # ret_concepts = convert_examples_to_ids(ret_concepts, self.concept2id) # seqlen
        return ret, ret_valid, ret_concepts
