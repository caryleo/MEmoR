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
    def __init__(self, vocab_size, embed_size, GAW=-1, concentration_factor=1):
        super(GraphAttention, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.concept_embed = nn.Embedding(vocab_size, embed_size)
        self.GAW = GAW if GAW >= 0 else None
        self.concentration_factor = concentration_factor
        ####################
        self.attn_concept = ScaledDotProductAttention((2 * self.embed_size) ** 0.5, attn_dropout=0)
        self.embed = nn.Embedding(vocab_size, embed_size)
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


    def get_hierarchical_sentence_representation(self, sent_embed):
        # sent_embed: (batch_size, seq_len, d_model)
        seq_len = sent_embed.shape[1]
        N = 3 # n-gram hierarchical pooling
        
        # max pooling for each ngram
        ngram_embeddings = [[] for i in range(N-1)] # one list for each n
        for n in range(1, N):
            for i in range(seq_len):
                ngram_embeddings[n-1].append(sent_embed[:,i:i+n+1,:].max(dim=1)[0])
        
        # mean pooling across ngram embeddings
        pooled_ngram_embeddings = [sent_embed.mean(dim=1)] # unigram
        for ngram_embedding in ngram_embeddings:
            ngram_embedding = torch.stack(ngram_embedding, dim=1).mean(dim=1)
            pooled_ngram_embeddings.append(ngram_embedding)

        sent_embed = torch.stack(pooled_ngram_embeddings, dim=1).mean(dim=1)
        return sent_embed

    def get_context_representation(self, src_embed, tgt_embed): 
        # src_embed: (batch_size, context_length*seq_len, d_model)
        # tgt_embed: (batch_size, seq_len, d_model)
        seq_len = tgt_embed.shape[1]
        context_length = src_embed.shape[1]//seq_len
        sentence_representations = []
        for i in range(context_length):
            sentence_representations.append(self.get_hierarchical_sentence_representation(src_embed[:, i*seq_len:(i+1)*seq_len]))
        sentence_representations.append(self.get_hierarchical_sentence_representation(tgt_embed))
        context_representation = torch.stack(sentence_representations, dim=1).mean(dim=1) # (batch_size, d_model)
        return context_representation

    def forward(self, src, src_embed, tgt, tgt_embed):
        # src: (batch_size, context_length * seq_len)
        # src_embed: (batch_size, context_length*seq_len, d_model)
        # tgt: (batch_size, seq_len)
        # tgt_embed: (batch_size, seq_len, d_model)
        # embed: shared embedding layer: (vocab_size, d_model)

        # get context representation
        context_representation = self.get_context_representation(src_embed, tgt_embed) # (batch_size, d_model)
        # get concept embedding
        src_len = src.shape[1]
        src = torch.cat([src, tgt], dim=1)
        cosine_similarity = torch.abs(torch.cosine_similarity(context_representation.unsqueeze(1), \
            self.concept_embed.weight.unsqueeze(0), dim=2)) # (batch_size, vocab_size)
        relatedness = self.edge_matrix[src] * cosine_similarity.unsqueeze(1) # (batch_size, (context_length+1)*seq_len, vocab_size)
        concept_weights = self._lambda*relatedness + (1-self._lambda)*(self.edge_matrix[src] > 0).float()*self.affectiveness # (batch_size, (context_length+1)*seq_len, vocab_size)
        concept_embedding = torch.matmul(torch.softmax(concept_weights * self.concentration_factor, dim=2), self.concept_embed.weight) # (batch_size, (context_length+1)*seq_len, d_model)
        return concept_embedding[:, :src_len, :], concept_embedding[:, src_len:, :]

    

