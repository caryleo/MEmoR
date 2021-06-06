import torch
from utils.util import read_json
from base import BaseFeatureExtractor

class AudioFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__()
        print("Initializing AudioFeatureExtracor...")
        self.feature_dim = config["audio"]["feature_dim"]
        self.feature_file = config["audio"]["feature_file"]
        self.data = read_json(config["data_file"])
        self.features = read_json(self.feature_file)
        ########
        # 音频概念的部分
        self.concepts = read_json(config["audio"]['concepts_file'])
        ########
        self.missing_tensor = torch.zeros((self.feature_dim))

    def get_feature(self, clip, target_character):
        on_characters = self.data[clip]["on_character"]
        if target_character not in on_characters:
            on_characters.append(target_character)
        on_characters = sorted(on_characters)
        speakers = self.data[clip]["speakers"]
        seg_ori_ind = self.data[clip]["seg_ori_ind"]
        ret = []
        ret_valid = []
        ##############
        ret_concepts = list()
        ##############

        for ii, speaker in enumerate(speakers): # seq_len, n_c
            index = "{}+{}".format(clip, seg_ori_ind[ii])
            for character in on_characters:
                if character == speaker:
                    if index in self.features:
                        ret.append(torch.tensor(self.features[index]))
                        ret_valid.append(1)
                    else:
                        ret.append(self.missing_tensor)
                        ret_valid.append(0)
                else:
                    ret.append(self.missing_tensor)
                    ret_valid.append(0)

            ret_concepts.append(self.concepts[index] if index in self.features else []) # seqlen

        # for character in on_characters:
        #     for ii, speaker in enumerate(speakers):
        #         index = "{}+{}".format(clip, seg_ori_ind[ii])
        #         if character == speaker:
                    
        #             if index in self.features:
        #                 ret.append(torch.tensor(self.features[index]))
        #                 ret_valid.append(1)
        #                 ###############
        #                 ret_concepts.append(self.concepts[index])
        #                 ###############
        #             else:
        #                 ret.append(self.missing_tensor)
        #                 ret_valid.append(0)
        #                 ###############
        #                 # 缺省状态，对应的就是空的标签列表
        #                 ret_concepts.append([])
        #                 ###############
        #         else:
        #             ret.append(self.missing_tensor)
        #             ret_valid.append(0)
        #             ###############
        #             # 缺省状态，对应的就是空的标签列表
        #             ret_concepts.append([])
        #             ###############

        ret = torch.stack(ret, dim=0) # seqlen * n_c, dim_feature_a
        ret_valid = torch.tensor(ret_valid, dtype=torch.int8) # seqlen * n_c
        # ret_concepts = convert_examples_to_ids(ret_concepts, self.concept2id) # seqlen
        return ret, ret_valid, ret_concepts 

