import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # some, 2 * dim_feature
        attn = torch.matmul(q / self.temperature, k.transpose(0, 1)) # some, some

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v) # some, 2 * dim_feature

        return output, attn


class GraphAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, config):
        super(GraphAttention, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.concept_embed = nn.Embedding(vocab_size, embed_size)
        self.GAW = config["model"]["args"]["graph_attention_weight"] if config["model"]["args"]["graph_attention_weight"] >= 0 else None
        self.concentration_factor = config["model"]["args"]["concentrator_factor"]
        ####################
        # self.attn_concept = ScaledDotProductAttention((2 * self.embed_size) ** 0.5, attn_dropout=0)
        self.embed = nn.Embedding(vocab_size, config["knowledge"]["embedding_dim"])
        ####################

    def init_params(self, edge_matrix=None, affectiveness=None, embedding_concept=None):
        edge_matrix_range = (edge_matrix.max(dim=1)[0] - edge_matrix.min(dim=0)[0]).unsqueeze(1)
        edge_matrix = edge_matrix/(edge_matrix_range + (edge_matrix_range==0).float()) # normalization
        self.edge_matrix = edge_matrix

        self.affectiveness = affectiveness
        
        if self.GAW is not None:
            self._lambda = self.GAW
        else:
            self._lambda = nn.Parameter(torch.full((self.vocab_size,), 0.5))

        self.embed.weight.data.copy_(torch.from_numpy(embedding_concept))


    def get_context_representation(self, concepts_embed, concepts_length): 
        # 将每一个上下文表示，包括当前本身，编码成一个上下文表示
        # 对于一组概念，暂时使用取平均的方式
        # 输入尺寸为seqlen, padlen, embed_size和seqlen
        # 预期输出尺寸为dim_representation

        context_rep_final = list()
        for i in range(concepts_embed.size(0)):
            context_rep_final.append(concepts_embed[i, :concepts_length[i]].mean(dim=0)) # embed_size

        context_representation = torch.stack(context_rep_final, dim=0).mean(dim=0) # dim

        return context_representation

    def forward(self, concepts_list, concepts_length):
        # 处理权重，范围一个等长度的seqlen, dim_representation的知识
        # 输入尺寸seqlen, padlen和seqlen

        concepts_embed = self.embed(concepts_list) # seqlen, pad_len, embed_size
        # get context representation
        context_representation = self.get_context_representation(concepts_embed, concepts_length) # embed_size
        
        # get concept embedding
        cosine_similarity = torch.abs(torch.cosine_similarity(context_representation.unsqueeze(0), \
            self.concept_embed.weight, dim=2)) # (vocab_size)

        # seqlen, padlen, vocab_size
        rel = self.edge_matrix[concepts_list] * cosine_similarity # seqlen, padlen, vocab_size
        aff = (self.edge_matrix[concepts_list] > 0).float() * self.affectiveness # seqlen, padlen, vocab_size
        concepts_weights = self._lambda * rel + (1 - self._lambda) * aff # seqlen, padlen, vocab_size
        concepts_embedding = torch.matmul(torch.softmax(concepts_weights * self.concentration_factor, dim=2), self.concept_embed.weight) # seqlen, padlen, dim_representation
        
        # 因为每一个片段只有一个特征，因此直接恢复回去，暂时不区分是否有效
        concepts_embedding_final = list()
        for i in range(concepts_list.size(0)):
            # 整合方法为平均
            concepts_embedding_final.append(concepts_embedding[i, :concepts_length[i]].mean(dim=0)) # dim_representation

        concepts_embedding = torch.stack(concepts_embedding_final, dim=0) # seqlen, dim_representation
        return concepts_embedding
    

