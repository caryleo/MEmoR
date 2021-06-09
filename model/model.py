from torch._C import Graph
import torch.nn as nn
import torch
import torch.nn.functional as F
from base import BaseModel
from .attention import GraphAttention, ScaledDotProductAttention


class AMER(BaseModel):
    def __init__(self, config):
        super().__init__()

    def initialize(self, config, device):

        n_classes = 9 if config['emo_type'] == 'primary' else 14
        D_e = config["model"]["args"]["D_e"]
        D_v = config["visual"]["dim_env"] + config["visual"]["dim_face"] + config["visual"]["dim_obj"]
        D_a = config["audio"]["feature_dim"]
        D_t = config["text"]["feature_dim"]
        D_p = config["personality"]["feature_dim"]
        
        self.attn = ScaledDotProductAttention((4 * D_e) ** 0.5, attn_dropout=0)

        self.enc_v = nn.Sequential(
            nn.Linear(D_v, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, D_e * 3),
            nn.ReLU(),
            nn.Linear(D_e * 3, 2 * D_e),
        )

        self.enc_a = nn.Sequential(
            nn.Linear(D_a, D_e * 8),
            nn.ReLU(),
            nn.Linear(D_e * 8, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )
        self.enc_t = nn.Sequential(
            nn.Linear(D_t, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )

        self.enc_p = nn.Sequential(
            nn.Linear(D_p, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )

        self.out_layer = nn.Sequential(
            nn.Linear(4 * D_e, 2 * D_e), 
            nn.ReLU(), 
            nn.Linear(2 * D_e, n_classes)
        )

        unified_d = 14 * D_e

        self.fusion_layer = nn.Linear(unified_d, 4 * D_e)

    def forward(self, U_v, U_a, U_t, U_p, M_v, M_a, M_t, seq_lengths, target_loc, seg_len, n_c):
        # Encoders
        V_e, A_e, T_e, P_e = self.enc_v(U_v), self.enc_a(U_a), self.enc_t(U_t), self.enc_p(U_p)

        U_all = []

        for i in range(M_v.shape[0]):
            target_moment, target_character = -1, -1
            for j in range(target_loc.shape[1]):
                if target_loc[i][j] == 1:
                    target_moment = j % int(seg_len[i].cpu().numpy())
                    target_character = int(j / seg_len[i].cpu().numpy())
                    break
            
            inp_V = V_e[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
            inp_T = T_e[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
            inp_A = A_e[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
            inp_P = P_e[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)

            mask_V = M_v[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)
            mask_T = M_t[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)
            mask_A = M_a[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)

            # Concat with personality embedding
            inp_V = torch.cat([inp_V, inp_P], dim=2)
            inp_A = torch.cat([inp_A, inp_P], dim=2)
            inp_T = torch.cat([inp_T, inp_P], dim=2)

            U = []

            for k in range(n_c[i]):
                new_inp_A, new_inp_T, new_inp_V = inp_A.clone(), inp_T.clone(), inp_V.clone(),
                
                # Modality-level inter-personal attention
                for j in range(seg_len[i]):
                    att_V, _ = self.attn(inp_V[j, :], inp_V[j, :], inp_V[j, :], mask_V[j, :])
                    att_T, _ = self.attn(inp_T[j, :], inp_T[j, :], inp_T[j, :], mask_T[j, :])
                    att_A, _ = self.attn(inp_A[j, :], inp_A[j, :], inp_A[j, :], mask_A[j, :])
                    new_inp_V[j, :] = att_V + inp_V[j, :]
                    new_inp_A[j, :] = att_A + inp_A[j, :]
                    new_inp_T[j, :] = att_T + inp_T[j, :]

                # Modality-level intra-personal attention
                att_V, _ = self.attn(new_inp_V[:, k], new_inp_V[:, k], new_inp_V[:, k], mask_V[:, k])
                att_A, _ = self.attn(new_inp_A[:, k], new_inp_A[:, k], new_inp_A[:, k], mask_A[:, k])
                att_T, _ = self.attn(new_inp_T[:, k], new_inp_T[:, k], new_inp_T[:, k], mask_T[:, k])

                # Residual connection
                inner_V = (att_V[target_moment] + new_inp_V[target_moment][k]).squeeze()
                inner_A = (att_A[target_moment] + new_inp_A[target_moment][k]).squeeze()
                inner_T = (att_T[target_moment] + new_inp_T[target_moment][k]).squeeze()

                # Multimodal fusion
                inner_U = self.fusion_layer(torch.cat([inner_V, inner_A, inner_T, inp_P[0][k]]))

                U.append(inner_U)

            if len(U) == 1:
                # Only one character in this sample
                U_all.append(U[0])
            else:
                # Person-level Inter-personal Attention
                U = torch.stack(U, dim=0)
                output, _ = self.attn(U, U, U)
                U = U + output
                U_all.append(U[target_character])

        U_all = torch.stack(U_all, dim=0)
        # Classification
        log_prob = self.out_layer(U_all)
        log_prob = F.log_softmax(log_prob)

        return log_prob


class AMER_Modi(BaseModel):
    def __init__(self, config):
        super().__init__()

    def initialize(self, config, device):

        n_classes = 9 if config['emo_type'] == 'primary' else 14
        D_e = config["model"]["args"]["D_e"]
        D_v = config["visual"]["dim_env"] + config["visual"]["dim_face"] + config["visual"]["dim_obj"]
        D_a = config["audio"]["feature_dim"]
        D_t = config["text"]["feature_dim"]
        D_p = config["personality"]["feature_dim"]

        #########################
        self.context_length = config["model"]["args"]["context_length"]
        #########################
        
        self.attn = ScaledDotProductAttention((4 * D_e) ** 0.5, attn_dropout=0)

        #################
        # GraphAttention初始化，暂时三个模态分开
        # 这里和模型的编码长度统一成256
        self.g_att_v = GraphAttention(config["visual"]["concept_size"], 2 * D_e, config)
        self.g_att_a = GraphAttention(config["audio"]["concept_size"], 2 * D_e, config)
        self.g_att_t = GraphAttention(config["text"]["concept_size"], 2 * D_e, config)
        #################

        self.enc_v = nn.Sequential(
            nn.Linear(D_v, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, D_e * 3),
            nn.ReLU(),
            nn.Linear(D_e * 3, 2 * D_e),
        )

        self.enc_a = nn.Sequential(
            nn.Linear(D_a, D_e * 8),
            nn.ReLU(),
            nn.Linear(D_e * 8, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )
        self.enc_t = nn.Sequential(
            nn.Linear(D_t, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )

        self.enc_p = nn.Sequential(
            nn.Linear(D_p, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )

        self.out_layer = nn.Sequential(
            nn.Linear(4 * D_e, 2 * D_e), 
            nn.ReLU(), 
            nn.Linear(2 * D_e, n_classes)
        )

        unified_d = 14 * D_e # 每种特征SA之后是4 De，和人格特征拼接是2 De

        self.fusion_layer = nn.Linear(unified_d, 4 * D_e)

    def forward(self, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C_v, C_a, C_t, concept_lengths_v, concept_lengths_a, concept_lengths_t, target_loc, seq_lengths, seg_len, n_c):
        # Encoders
        V_e, A_e, T_e, P_e = self.enc_v(U_v), self.enc_a(U_a), self.enc_t(U_t), self.enc_p(U_p) # batch, seq_len * n_c, dim_feature

        U_all = []

        # 来自原始实现：因为每一个样本都有点复杂，所以一个一个样本的计算，然后汇到一块进行loss计算
        for i in range(M_v.shape[0]):
            # 从每一个样本进行当前样本位置判定
            target_moment, target_character = -1, -1
            for j in range(target_loc.shape[1]):
                if target_loc[i][j] == 1: # 因为原始的顺序从n_c, seqlen调成了seqlen, n_c，行对应序列，列对应角色
                    target_moment = int(j / seg_len[i].cpu().numpy())
                    target_character = j % int(seg_len[i].cpu().numpy())
                    break

            ################### Graph Attetion 处理位点 ################################
            # 上下文信息
            kecr_V = self.g_att_v(C_v[i], concept_lengths_v[i]) # seqlen, dim_representation
            kecr_A = self.g_att_a(C_a[i], concept_lengths_a[i]) # seqlen, dim_representation
            kecr_T = self.g_att_t(C_t[i], concept_lengths_t[i]) # seqlen, dim_representation 
            ###########################################################################

            # 特征的尺寸都是统一的2 De
            inp_V = V_e[i, : seq_lengths[i], :].reshape((seg_len[i], n_c[i], -1)) # seq_len, n_c, dim_feature
            inp_A = A_e[i, : seq_lengths[i], :].reshape((seg_len[i], n_c[i], -1)) # seq_len, n_c, dim_feature
            inp_T = T_e[i, : seq_lengths[i], :].reshape((seg_len[i], n_c[i], -1)) # seq_len, n_c, dim_feature
            inp_P = P_e[i, : seq_lengths[i], :].reshape((seg_len[i], n_c[i], -1)) # seq_len, n_c, dim_feature

            mask_V = M_v[i, : seq_lengths[i]].reshape((seg_len[i], n_c[i])) # seq_len, n_c
            mask_A = M_a[i, : seq_lengths[i]].reshape((seg_len[i], n_c[i])) # seq_len, n_c
            mask_T = M_t[i, : seq_lengths[i]].reshape((seg_len[i], n_c[i])) # seq_len, n_c

            #################################
            # Concat with personality embedding 个性化人物特征拼接，知识特征插入位点
            inp_V = torch.cat([inp_V, kecr_V.unsqueeze(1).expand(-1, inp_V.size(1), -1), inp_P], dim=2) # seq_len, n_c, 2 * dim_feature + dim_representation
            inp_A = torch.cat([inp_A, kecr_A.unsqueeze(1).expand(-1, inp_A.size(1), -1), inp_P], dim=2) # seq_len, n_c, 2 * dim_feature + dim_representation
            inp_T = torch.cat([inp_T, kecr_T.unsqueeze(1).expand(-1, inp_T.size(1), -1), inp_P], dim=2) # seq_len, n_c, 2 * dim_feature + dim_representation
            #################################
            
            U = []

            for k in range(n_c[i]):
                new_inp_A, new_inp_T, new_inp_V = inp_A.clone(), inp_T.clone(), inp_V.clone() # seq_len, n_c, 2 * dim_feature
                
                # Modality-level inter-personal attention
                for j in range(seg_len[i]):
                    # 先是每一个时刻，角色间的上下文建模
                    att_V, _ = self.attn(inp_V[j, :], inp_V[j, :], inp_V[j, :], mask_V[j, :]) # n_c, 2 * dim_feature
                    att_A, _ = self.attn(inp_A[j, :], inp_A[j, :], inp_A[j, :], mask_A[j, :]) # n_c, 2 * dim_feature
                    att_T, _ = self.attn(inp_T[j, :], inp_T[j, :], inp_T[j, :], mask_T[j, :]) # n_c, 2 * dim_feature

                    # Residual connection
                    new_inp_V[j, :] = att_V + inp_V[j, :] # seq_len, n_c, 2 * dim_feature
                    new_inp_A[j, :] = att_A + inp_A[j, :] # seq_len, n_c, 2 * dim_feature
                    new_inp_T[j, :] = att_T + inp_T[j, :] # seq_len, n_c, 2 * dim_feature

                # Modality-level intra-personal attention
                # 接着是每一个角色，所有的时刻
                att_V, _ = self.attn(new_inp_V[:, k], new_inp_V[:, k], new_inp_V[:, k], mask_V[:, k]) # seq_len, 2 * dim_feature
                att_A, _ = self.attn(new_inp_A[:, k], new_inp_A[:, k], new_inp_A[:, k], mask_A[:, k]) # seq_len, 2 * dim_feature
                att_T, _ = self.attn(new_inp_T[:, k], new_inp_T[:, k], new_inp_T[:, k], mask_T[:, k]) # seq_len, 2 * dim_feature

                # Residual connection
                inner_V = (att_V[target_moment] + new_inp_V[target_moment][k]).squeeze() # 2 * dim_feature
                inner_A = (att_A[target_moment] + new_inp_A[target_moment][k]).squeeze() # 2 * dim_feature
                inner_T = (att_T[target_moment] + new_inp_T[target_moment][k]).squeeze() # 2 * dim_feature

                # Multimodal fusion
                # 多模态特征融合，可考虑改动
                inner_U = self.fusion_layer(torch.cat([inner_V, inner_A, inner_T, inp_P[0][k]])) # 2 * dim_feature

                U.append(inner_U)

            if len(U) == 1:
                # Only one character in this sample
                U_all.append(U[0])
            else:
                # Person-level Inter-personal Attention
                U = torch.stack(U, dim=0) # n_c, 2 * dim_feature
                output, _ = self.attn(U, U, U) # n_c, 2 * dim_feature
                U = U + output # n_c, 2 * dim_feature
                U_all.append(U[target_character]) # 2 * dim_feature

        U_all = torch.stack(U_all, dim=0) # batch, 2 * dim_feature
        # Classification
        log_prob = self.out_layer(U_all) # batch, 2 * dim_feature
        log_prob = F.log_softmax(log_prob) # batch, n_classes

        return log_prob # batch, n_classes
