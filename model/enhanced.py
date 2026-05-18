import torch
from torch import nn
import torch.nn.functional as F

class KG_enhanced(nn.Module):
    def __init__(self,E, R, hidden_dim, head_num=1, R_dim=512, attention_dim=128, gamma=18):
        super().__init__()
        ## 0. 공통 정의 E(entity 개수) R(relation 개수) 
        # E_dim(ent embedding의 hidden dim) R_dim(rel embedding의 hidden dim) 
        # H(LLM의 hidden embedding) A(attention의 hidden embedding) 
        self.E = E
        self.R = R
        self.R_dim = R_dim
        self.E_dim = R_dim * 2
        self.H = hidden_dim
        self.head_num = head_num
        self.A = attention_dim
        # E_e, E_r 정의
        self.epsilon = 2.0
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.hidden_dim]), 
            requires_grad=False
        )
        self.entity_embedding = nn.Parameter(torch.zeros(self.E, self.E_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        self.relation_embedding = nn.Parameter(torch.zeros(self.R, self.R_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        ## 1. KGC loss 를 위한 W_q, W_k, W_v 정의
        self.W_q = nn.Linear(self.E_dim, self.A, bias=False)
        self.W_k = nn.Linear(self.H, self.A, bias=False)
        self.W_v = nn.Linear(self.H, self.E_dim, bias=False)
        ## 2. LLM loss 를 위한 W_s 정의
        self.W_s = nn.Linear(self.H, self.E_dim, bias=False)
    
    def KGC_loss():
        pass

    def LLM_loss():
        pass 

    def forward(self): 
        #, x_context, extract_pos, attn_mask, query_ids, entity_ids, triple_ids, is_predicted_tail, subgraph):
        if x_context.dim() == 3:
            x = x_context[extract_pos[:, 0], extract_pos[:, 1]]
            x_attn = x_context
        else:
            x = x_context
            x_attn = None

        kgc_loss = self.KGC_loss()
        llm_loss = self.LLM_loss() 
        total_loss = kgc_loss + llm_loss
        return {
            "total_loss": total_loss,
            "kgc_loss": kgc_loss,
            "llm_loss": llm_loss
        }