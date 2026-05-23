import torch
from torch import nn
import torch.nn.functional as F
import math 

class KG_enhanced(nn.Module):
    def __init__(self,E, R, hidden_dim, rand_neg, KGE_model_name='TransE', head_num=1, R_dim=1000, attention_dim=1024, gamma=18, num_neg_samples = 256):
        super().__init__()
        ## 0. 공통 정의 E(entity 개수) R(relation 개수) 
        # E_dim(ent embedding의 hidden dim) R_dim(rel embedding의 hidden dim) 
        # H(LLM의 hidden embedding) A(attention의 hidden embedding)
        self.KGE_model = KGE_model_name 
        if self.KGE_model.lower() == 'rotate':
            self.dim_different = True
        elif self.KGE_model.lower() == 'transe':
            self.dim_different = False
        else:
            raise ValueError(f"not prepared for the self.KGE_model:{self.KGE_model}")
        self.E = E
        self.R = R
        self.R_dim = R_dim
        self.E_dim = R_dim * 2 if self.dim_different else R_dim
        
        self.H = hidden_dim
        self.head_num = head_num
        self.A = attention_dim
        # E_e, E_r 정의
        self.epsilon = 2.0
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.H]), 
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
        if self.dim_different:
            self.W_q_r = nn.Linear(self.R_dim, self.A, bias=False)
            self.W_o_r = nn.Linear(self.E_dim, self.R_dim, bias=False)
            nn.init.zeros_(self.W_o_r.weight)
        ## 2. LLM loss 를 위한 W_s, tau 정의
        self.W_s = nn.Linear(self.E_dim, self.H, bias=False)
        self.tau = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        ## 3. hyper-parameter
        self.num_neg_samples = num_neg_samples
        self.random_neg = rand_neg 
    
    def _apply_attention(self, emb, K, V, attention_mask, is_relation=False):
        """
        emb: (B,E_dim) or (B,20,E_dim)
        K: (B,L,A)
        V: (B,L,E_dim)
        attention_mask: (B,L)
        """
        is_2d = (emb.dim()==2)
        if is_2d:
            emb = emb.unsqueeze(1) # (B,1,E_dim)
        if self.dim_different and is_relation:
            Q = self.W_q_r(emb) # (B,1,A)
        else:
            Q = self.W_q(emb) # (B,1,A) or (B,20,A)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.A) # (B,1,L) or (B,20,L)
        extended_mask = attention_mask.unsqueeze(1) #(B,1,L)
        scores = scores.masked_fill(extended_mask==0, -1e9) # (B,1,L) or (B,20,L)
        attn_weights = F.softmax(scores, dim=-1) # (B,1,L) or (B,20,L)
        context = torch.matmul(attn_weights, V) #(B,1,E_dim) or (B,20,E_dim)
        if self.dim_different and is_relation:
            context = self.W_o_r(context) #(B,1,R_dim)
        emb_tilde = emb + context 
        if is_2d:
            emb_tilde = emb_tilde.squeeze(1) # (B,E_dim)
        return emb_tilde

    def _calculate_KGE_distance(self, h_fixed, r_fixed, t_fixed, cand_H, cand_T):
        if self.KGE_model.lower() == "rotate":
            pi = 3.14159265358979323846
            #head, tail은 a+bi의 a,b를 그대로 임베딩에서 가져오고, relation은 임베딩 값 x \in R 일때, x를 각도로 봄. -> cos,sin 계산 
            re_head, im_head = torch.chunk(h_fixed, 2, dim=-1) # h_fixed \in R^(8,4096) # re_head \in R^(8,2048)
            re_tail, im_tail = torch.chunk(t_fixed, 2, dim=-1)
            re_cand_H, im_cand_H = torch.chunk(cand_H, 2, dim=-1) # cand_H \in R^(8,276,4096) # re_cand_H \in R^(8,276,2048)
            re_cand_T, im_cand_T = torch.chunk(cand_T, 2, dim=-1)
            phase_relation = r_fixed / (self.embedding_range.item()/pi)
            re_relation = torch.cos(phase_relation)
            im_relation = torch.sin(phase_relation)
            # if mode == "head-batch": #이게 |t(~r)-h| # ~는 켤레복소수기호 
            re_score_head = (re_relation * re_tail + im_relation * im_tail).unsqueeze(1)
            im_score_head = (re_relation * im_tail - im_relation * re_tail).unsqueeze(1)
            re_score_head = re_score_head - re_cand_H # (8,276,2048) = (8,1,2048) - (8,276,2048)
            im_score_head = im_score_head - im_cand_H
            score_head = torch.stack([re_score_head,im_score_head], dim=0) # (2,8,276,2048)
            d_head = score_head.norm(dim=0).sum(dim=2) # (8,276,2048).(8,276)
            # else:#이게 |hr-t|
            re_score_tail = (re_head * re_relation - im_head * im_relation).unsqueeze(1)
            im_score_tail = (re_head * im_relation + im_head * re_relation).unsqueeze(1)
            re_score_tail = re_score_tail - re_cand_T 
            im_score_tail = im_score_tail - im_cand_T
            score_tail = torch.stack([re_score_tail,im_score_tail], dim=0)
            d_tail = score_tail.norm(dim=0).sum(dim=2)
        elif self.KGE_model.lower() == 'transe':
            # margin-based loss에서는 L1 norm 사용.
            q_tail = (h_fixed + r_fixed).unsqueeze(1) # (8,1,E_dim)
            d_tail = torch.norm(q_tail - cand_T, p=1, dim=2) # (8,1,E_dim)-(8,20,E_dim)->(8,20) dim=2라서 

            q_head = (t_fixed - r_fixed).unsqueeze(1)
            d_head = torch.norm(cand_H - q_head, p=1, dim=2)
        else:
            raise RuntimeError(f"[에러 감지] 지원하지 않는 model_type이 들어왔습니다: '{self.KGE_model}'")
        return d_head, d_tail 

    def _compute_structure_embedding(self, distances, cand_embs):
        """
        distances: (B, num_cands)
        cand_embs: (B, num_cands, E_dim)
        """
        scaled_distances = -distances / self.tau # [8,20]
        p = F.softmax(scaled_distances, dim=-1)  # [8,20]
        p = p.unsqueeze(1) # [8,1,20]
        S = torch.bmm(p, cand_embs).squeeze(1) # [8,1,500] -> [8,500]
        S = self.W_s(S) # [8,4096] (W_s:[500,4096])
        return S

    def KGC_loss(self, last_hidden_states, attention_mask, triple_ids, entity_ids, is_predicted_tail, adv_temperature=1.0):
        batch_size = entity_ids.size(0)
        # Key, Value 준비 
        K = self.W_k(last_hidden_states) # (B,L,A) = (H,A) (B,L,H)
        V = self.W_v(last_hidden_states) # (B,L,E_dim)
        # Query 준비 # triple_ids [8,3] 
        ## head
        head_emb = self.entity_embedding[triple_ids[:,0]] #(B,E_dim)
        head_emb_tilde = self._apply_attention(head_emb, K, V, attention_mask)
        ## tail
        tail_emb = self.entity_embedding[triple_ids[:,2]] #(B,E_dim)
        tail_emb_tilde = self._apply_attention(tail_emb, K, V, attention_mask)
        ## cand
        num_real_cands = entity_ids.size(1) # 원래 후보 개수 (20)
        if self.random_neg:
            # Negative sampling 추가 (위로 이동) # 일단 256, 하이퍼파라미터 #num_neg_samples = 256
            rand_entity_ids = torch.randint(0, self.E, (batch_size, self.num_neg_samples), device=entity_ids.device)
            # 기존 20개 + 랜덤 256개 병합 -> 총 276개의 후보군
            combined_ids = torch.cat([entity_ids, rand_entity_ids], dim=1)
        else:
            combined_ids = entity_ids
        cand_emb = self.entity_embedding[combined_ids] #(B,276,E_dim) or (B,20,E_dim)
        cand_emb_tilde = self._apply_attention(cand_emb, K, V, attention_mask)
        ## rel
        rel_emb = self.relation_embedding[triple_ids[:,1]] #(B,E_dim/2)
        rel_emb_tilde = self._apply_attention(rel_emb, K, V, attention_mask, is_relation=True)
        # d_i 구하기 # 여기까지는 동일

        d_head, d_tail = self._calculate_KGE_distance(head_emb_tilde, rel_emb_tilde, tail_emb_tilde, cand_emb_tilde, cand_emb_tilde)

        is_tail = is_predicted_tail.unsqueeze(1) # [8,1]
        distances = torch.where(is_tail, d_tail, d_head) # [8,20]
        
        real_distances = distances[:, :num_real_cands]       # (B, 20)
        real_cand_embs = cand_emb_tilde[:, :num_real_cands, :] # (B, 20, E_dim)

        # 함수 호출! S 리턴받기
        structure_embedding = self._compute_structure_embedding(real_distances, real_cand_embs) # (B, H)

        target_entity = torch.where(is_predicted_tail, triple_ids[:, 2], triple_ids[:, 0]) # (B)
        matches = (combined_ids == target_entity.unsqueeze(1)) # (B,20) True, False로 이루어짐 

        # 정답 거리 (d_pos) 추출
        # [주의] 랜덤 샘플링 중 정답이 우연히 1번 더 뽑혀서 matches가 True인 곳이 2개 이상일 수 있음. 
        # 중복 합산을 막기 위해 True의 개수로 나눠 평균 처리.
        num_matches = matches.float().sum(dim=1, keepdim=True).clamp(min=1.0) #[8,1]
        d_pos = (distances * matches.float()).sum(dim=1, keepdim=True) / num_matches

        # Margin 점수 변환 (Score = Gamma - Distance)
        gamma = self.gamma.item()
        pos_score = gamma - d_pos # [8,1]
        neg_score = gamma - distances # 오답을 포함한 전체 후보군의 점수 # [8,20]

        # 정답(Positive) Loss 계산
        pos_loss = -F.logsigmoid(pos_score).squeeze(dim=1) # [batch_size]

        # ---------------------------------------------------------
        # 적대적 오답 샘플링 (Negative Adversarial Sampling) 적용
        # ---------------------------------------------------------
        # (1) Softmax를 오답에만 적용하기 위해, 진짜 정답 위치의 점수를 매우 낮게(-1e9) 마스킹
        adv_neg_score = neg_score.clone()
        adv_neg_score[matches] = -1e9 

        # (2) 오답들에 대한 적대적 가중치 계산
        # 점수가 높을수록(거리가 가까워서 정답으로 착각하기 쉬울수록) 높은 가중치를 가짐
        # .detach()를 적용하여 가중치 자체에는 역전파(Backprop)가 되지 않도록 함 (Self-Adversarial 논문 원칙)
        neg_weights = F.softmax(adv_neg_score * adv_temperature, dim=1).detach()

        # (3) 가중치가 반영된 오답(Negative) Loss 계산
        # Softmax 특성상 neg_weights의 합이 1이므로, 이전처럼 오답 개수로 평균 낼 필요 없이 바로 sum()
        neg_loss_all = neg_weights * (-F.logsigmoid(-neg_score))
        neg_loss = neg_loss_all.sum(dim=1) # [batch_size]

        # 10. 최종 Loss (Positive와 Negative의 평균)
        loss = (pos_loss.mean() + neg_loss.mean()) / 2.0
        return structure_embedding, loss

    def forward(self, lhs_cut, attn_mask, triple_ids, entity_ids, is_predicted_tail):
        #breakpoint()
        structure_embedding, kgc_loss = self.KGC_loss(lhs_cut, attn_mask, triple_ids, entity_ids, is_predicted_tail)
        return structure_embedding, kgc_loss