import torch
from torch import nn
import torch.nn.functional as F
import math 

class KG_align(nn.Module):
    def __init__(self,E, R, hidden_dim, rand_neg, beta, pretrained_ent=None, pretrained_rel=None, freeze_embeddings=True, KGE_model_name='TransE', head_num=1, R_dim=1000, attention_dim=1024, gamma=18, num_neg_samples = 256):
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
        #breakpoint()
        self.freeze_embeddings = freeze_embeddings
        if pretrained_ent is not None:
            clean_ent = pretrained_ent.detach().clone()
            self.entity_embedding = nn.Parameter(clean_ent, requires_grad=(not self.freeze_embeddings))
        else:
            self.entity_embedding = nn.Parameter(torch.zeros(self.E, self.E_dim))
            nn.init.uniform_(
                tensor=self.entity_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        if pretrained_rel is not None:
            clean_rel = pretrained_rel.detach().clone()
            self.relation_embedding = nn.Parameter(clean_rel, requires_grad=(not self.freeze_embeddings))
        else:
            self.relation_embedding = nn.Parameter(torch.zeros(self.R, self.R_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        ## 1. KGC loss 를 위한 W_q, W_k, W_v 정의
        self.W_q = nn.Linear(self.H, self.E_dim, bias=False)
        if self.dim_different:
            self.W_q_r = nn.Linear(self.R_dim, self.A, bias=False)
            self.W_o_r = nn.Linear(self.E_dim, self.R_dim, bias=False)
            nn.init.zeros_(self.W_o_r.weight)
        ## 2. LLM loss 를 위한 W_s, tau 정의
        self.tau = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        ## 3. hyper-parameter
        self.num_neg_samples = num_neg_samples
        self.random_neg = rand_neg 
        self.beta = beta

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

    def KGE_loss(self, triple_ids, entity_ids, is_predicted_tail, rand_entity_ids, adv_temperature=1.0):
        ## head
        head_emb = self.entity_embedding[triple_ids[:,0]] #(B,E_dim)
        ## tail
        tail_emb = self.entity_embedding[triple_ids[:,2]] #(B,E_dim)
        ## cand
        num_real_cands = entity_ids.size(1) # 원래 후보 개수 (20)
        if self.random_neg:
            combined_ids = torch.cat([entity_ids, rand_entity_ids], dim=1)
        else:
            combined_ids = entity_ids
        cand_emb = self.entity_embedding[combined_ids] #(B,276,E_dim) or (B,20,E_dim)
        ## rel
        rel_emb = self.relation_embedding[triple_ids[:,1]] #(B,E_dim/2)
        # d_i 구하기 # 여기까지는 동일
        
        # kge loss
        d_head, d_tail = self._calculate_KGE_distance(head_emb, rel_emb, tail_emb, cand_emb, cand_emb)

        is_tail = is_predicted_tail.unsqueeze(1) # [8,1]
        distances = torch.where(is_tail, d_tail, d_head) # [8,20]

        target_entity = torch.where(is_predicted_tail, triple_ids[:, 2], triple_ids[:, 0]) # (B)
        matches = (combined_ids == target_entity.unsqueeze(1)) #(8,276)==(8,1) True, False로 이루어짐 

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
        safe_min = torch.finfo(adv_neg_score.dtype).min
        adv_neg_score[matches] = safe_min

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
        return loss

    def Align_loss(self, last_hidden_state, triple_ids, entity_ids, is_predicted_tail, rand_entity_ids,adv_temperature=1.0):
        batch_size = entity_ids.size(0)
        #q = A h_x
        query = self.W_q(last_hidden_state) # [8, 500]
        ## head
        head_emb = self.entity_embedding[triple_ids[:,0]] #(B,E_dim)
        ## tail
        tail_emb = self.entity_embedding[triple_ids[:,2]] #(B,E_dim)
        ## cand
        if self.random_neg:
            combined_ids = torch.cat([entity_ids, rand_entity_ids], dim=1)
        else:
            combined_ids = entity_ids
        cand_emb = self.entity_embedding[combined_ids] # [8, 276, 500]
        ## rel
        rel_emb = self.relation_embedding[triple_ids[:,1]] #(B,E_dim/2)
        # d_i 구하기 # 여기까지는 동일

        d_query = query.unsqueeze(1) # [B,1,E_dim]
        distances = torch.norm(d_query - cand_emb, p=1, dim=2) # [8, 276]

        is_tail = is_predicted_tail.unsqueeze(1) # [8,1]
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
        safe_min = torch.finfo(adv_neg_score.dtype).min
        adv_neg_score[matches] = safe_min

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
        return loss

    def forward(self, last_hidden_state, triple_ids, entity_ids, is_predicted_tail, is_infer):
        last_hidden_state = last_hidden_state.float()
        if is_infer:# infer 모든 t에 대해 ranking 
            # (1) V_final 구하기 
            query = self.W_q(last_hidden_state).float() # [1,4096]
            all_entities = self.entity_embedding.float()
            rel_emb = self.relation_embedding[triple_ids[:,1]].float() # [1,500]
            is_tail = is_predicted_tail.item() if isinstance(is_predicted_tail, torch.Tensor) else is_predicted_tail
            with torch.cuda.amp.autocast(enabled=False):
                if is_tail: # head, relation embedding 구하기 -> \delta
                    head_emb = self.entity_embedding[triple_ids[:,0]].float()
                    temp = head_emb + rel_emb
                else: # relation, tail embedding 구하기 -> \delta
                    tail_emb = self.entity_embedding[triple_ids[:,2]].float()
                    temp = tail_emb - rel_emb
            delta = torch.norm(query - temp, p=1, dim=-1, keepdim=True) # [1]
            alpha = torch.exp(- self.beta * delta) # beta 2개 고르기 +arg에 추가 
            V_final = alpha * query + (1-alpha) * temp # [1,500]
            #V_final = temp ## alpha=0으로 고정 
            score_tensor = V_final.unsqueeze(1) - all_entities.unsqueeze(0)
            distances = torch.norm(score_tensor, p=1, dim=2)
            scores = -distances
            return scores
        else:
            batch_size = entity_ids.size(0)
            rand_entity_ids = torch.randint(0, self.E, (batch_size, self.num_neg_samples), device=entity_ids.device)
            if self.freeze_embeddings:
                kge_loss = torch.tensor(0.0, device=entity_ids.device)
            else:
                kge_loss = self.KGE_loss(triple_ids, entity_ids, is_predicted_tail, rand_entity_ids)
            align_loss = self.Align_loss(last_hidden_state, triple_ids, entity_ids, is_predicted_tail, rand_entity_ids)
            return {
                "align_loss": align_loss,
                "kge_loss": kge_loss
            }
