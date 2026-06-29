import torch
from torch import nn
import torch.nn.functional as F
import math 

class KG_align(nn.Module):
    def __init__(self,E, R, hidden_dim, rand_neg, alpha, beta, use_d_r, pretrained_ent=None, pretrained_rel=None, freeze_embeddings=True, KGE_model_name='TransE', head_num=1, R_dim=1000, attention_dim=1024, gamma=18, num_neg_samples = 256):
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
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.R_dim]), 
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
        ## 방법2: 공유 matrix 하나, relation-specific vector 하나 (q = (Ah_x) \circ d_r)
        if use_d_r:
            self.d_r = nn.Parameter(torch.zeros(self.R, self.E_dim))
            nn.init.uniform_(
                tensor=self.d_r,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        # if self.dim_different:
        #     self.W_q_r = nn.Linear(self.R_dim, self.A, bias=False)
        #     self.W_o_r = nn.Linear(self.E_dim, self.R_dim, bias=False)
        #     nn.init.zeros_(self.W_o_r.weight)
        ## 2. LLM loss 를 위한 W_s, tau 정의
        self.tau = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        ## 3. hyper-parameter
        self.num_neg_samples = num_neg_samples
        self.random_neg = rand_neg 
        self.alpha = alpha
        self.beta = beta
        self.use_d_r = use_d_r
    
    def _get_target_point(self, ent_emb, rel_emb, is_tail):
        """
        [Atomic 함수 1] 주어진 엔티티와 관계로 목표점(Target Point) 텐서를 반환합니다.
        - is_tail=True (Tail 예측): h + r  또는 h \circ r
        - is_tail=False (Head 예측): t - r 또는 t \circ r^{-1}
        """
        if self.KGE_model.lower() == 'rotate':
            pi = 3.14159265358979323846
            phase_relation = rel_emb / (self.embedding_range.item() / pi)
            re_rel = torch.cos(phase_relation)
            im_rel = torch.sin(phase_relation)
            re_ent, im_ent = torch.chunk(ent_emb, 2, dim=-1)
            
            if is_tail: # h * r
                re_target = re_ent * re_rel - im_ent * im_rel
                im_target = re_ent * im_rel + im_ent * re_rel
            else: # t * r^{-1} (복소수 켤레 곱셈)
                re_target = re_ent * re_rel + im_ent * im_rel
                im_target = im_ent * re_rel - re_ent * im_rel
            return torch.cat([re_target, im_target], dim=-1)
            
        elif self.KGE_model.lower() == 'transe': # TransE
            if is_tail:
                return ent_emb + rel_emb
            else:
                return ent_emb - rel_emb
        else:
            raise RuntimeError(f"[에러 감지] 지원하지 않는 model_type이 들어왔습니다: '{self.KGE_model}'")

    def _calc_distance(self, v1, v2, keepdim=False):
        """
        [Atomic 함수 2] 두 텐서 v1, v2 사이의 KGE 거리를 계산합니다.
        """
        if self.KGE_model.lower() == 'rotate':
            re_v1, im_v1 = torch.chunk(v1, 2, dim=-1)
            re_v2, im_v2 = torch.chunk(v2, 2, dim=-1)
            # 실수부 차이, 허수부 차이를 구한 뒤 복소수 크기(Norm) 계산 후 차원 합산
            diff_re = re_v1 - re_v2
            diff_im = im_v1 - im_v2
            dist = torch.stack([diff_re, diff_im], dim=0).norm(dim=0).sum(dim=-1, keepdim=keepdim)
            return dist
        if self.KGE_model.lower() == 'transe': # TransE
            return torch.norm(v1 - v2, p=1, dim=-1, keepdim=keepdim)
        else:
            raise RuntimeError(f"[에러 감지] 지원하지 않는 model_type이 들어왔습니다: '{self.KGE_model}'")
    
    def _calculate_KGE_distance(self, h_fixed, r_fixed, t_fixed, cand_H, cand_T):
        # 1. Tail 예측 거리 계산 (|h + r - cand_T|)
        target_tail = self._get_target_point(h_fixed, r_fixed, is_tail=True).unsqueeze(1)
        d_tail = self._calc_distance(target_tail, cand_T)

        # 2. Head 예측 거리 계산 (|cand_H + r - t| -> |cand_H - (t - r)|)
        target_head = self._get_target_point(t_fixed, r_fixed, is_tail=False).unsqueeze(1)
        d_head = self._calc_distance(cand_H, target_head) 

        return d_head, d_tail
    
    def _calculate_KGE_distance2(self, h_fixed, r_fixed, t_fixed, cand_H, cand_T):
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
        #d_head2, d_tail2 = self._calculate_KGE_distance2(head_emb, rel_emb, tail_emb, cand_emb, cand_emb) # new _calculate_KGE_distance 
        #breakpoint()
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

    def Align_loss(self, last_hidden_state, triple_ids, entity_ids, is_predicted_tail, rand_entity_ids, adv_temperature=1.0):
        batch_size = entity_ids.size(0)
        #q = A h_x
        Ah_x = self.W_q(last_hidden_state) # [8, 500]
        if self.use_d_r:
            d_r = self.d_r[triple_ids[:,1]]
            query = Ah_x *d_r
        else:
            query = Ah_x
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
    
    def Structure_loss(self, last_hidden_state, triple_ids, is_predicted_tail):
        Ah_x = self.W_q(last_hidden_state) # [8, 500]
        if self.use_d_r:
            d_r = self.d_r[triple_ids[:,1]]
            query = Ah_x *d_r
        else:
            query = Ah_x
        head_emb = self.entity_embedding[triple_ids[:,0]] #(B,E_dim)
        rel_emb = self.relation_embedding[triple_ids[:,1]] 
        tail_emb = self.entity_embedding[triple_ids[:,2]] #(B,E_dim)
        is_tail = is_predicted_tail.unsqueeze(1) # [B, 1]
        query_cand = query.unsqueeze(1) # [B, 1, E_dim]
        d_head, d_tail = self._calculate_KGE_distance(
            head_emb, rel_emb, tail_emb, 
            cand_H=query_cand, cand_T=query_cand
        )
        distances = torch.where(is_tail, d_tail, d_head) # [B, 1]
        loss = distances.mean()
        return loss


    def forward(self, last_hidden_state, triple_ids, entity_ids, is_predicted_tail, is_infer):
        last_hidden_state = last_hidden_state.float()
        #breakpoint()
        if is_infer:# infer 모든 t에 대해 ranking 
            # (1) V_final 구하기 
            Ah_x = self.W_q(last_hidden_state).float() # [8, 500]
            if self.use_d_r:
                d_r = self.d_r[triple_ids[:,1]].float()
                query = Ah_x *d_r
            else:
                query = Ah_x
            #query = self.W_q(last_hidden_state).float() # [1,4096]
            all_entities = self.entity_embedding.float()
            rel_emb = self.relation_embedding[triple_ids[:,1]].float() # [1,500]
            is_tail = is_predicted_tail.item() if isinstance(is_predicted_tail, torch.Tensor) else is_predicted_tail
            with torch.cuda.amp.autocast(enabled=False):
                # 🌟 1. Target Point (temp) 구하기 (Atomic 1 호출)
                fixed_ent_idx = 0 if is_tail else 2
                fixed_ent = self.entity_embedding[triple_ids[:, fixed_ent_idx]].float()
                temp = self._get_target_point(fixed_ent, rel_emb, is_tail)

                # 🌟 2. Query와 Temp 사이의 거리(delta) 구하기 (Atomic 2 호출)
                delta = self._calc_distance(query, temp, keepdim=True)
                
                # 🌟 3. V_final 병합 연산
                if self.beta > 0:
                    alpha_weight = torch.exp(- self.beta * delta)
                    V_final = alpha_weight * query + (1 - alpha_weight) * temp 
                else:
                    V_final = self.alpha * query + (1 - self.alpha) * temp 
                
                # 🌟 4. 전체 엔티티에 대한 최종 랭킹 거리 구하기 (Atomic 2 재활용)
                distances = self._calc_distance(V_final.unsqueeze(1), all_entities.unsqueeze(0))
                scores = -distances
                
                return scores
                if is_tail: # head, relation embedding 구하기 -> \delta
                    head_emb = self.entity_embedding[triple_ids[:,0]].float()
                    temp = head_emb + rel_emb
                else: # relation, tail embedding 구하기 -> \delta
                    tail_emb = self.entity_embedding[triple_ids[:,2]].float()
                    temp = tail_emb - rel_emb
            delta = torch.norm(query - temp, p=1, dim=-1, keepdim=True) # [1]
            if self.beta > 0:
                alpha = torch.exp(- self.beta * delta) # beta 2개 고르기 +arg에 추가 
                V_final = alpha * query + (1-alpha) * temp # [1,500]
            else:
                V_final = self.alpha * query + (1-self.alpha)*temp 
            score_tensor = V_final.unsqueeze(1) - all_entities.unsqueeze(0)
            distances = torch.norm(score_tensor, p=1, dim=2)
            scores = -distances
            #breakpoint() # torch.allclose(socre, score2, atol=1e-6) # True 나옴 확인 
            return scores
        else:
            #breakpoint()
            batch_size = entity_ids.size(0)
            rand_entity_ids = torch.randint(0, self.E, (batch_size, self.num_neg_samples), device=entity_ids.device)
            kge_loss = self.KGE_loss(triple_ids, entity_ids, is_predicted_tail, rand_entity_ids)
            align_loss = self.Align_loss(last_hidden_state, triple_ids, entity_ids, is_predicted_tail, rand_entity_ids)
            struct_loss = self.Structure_loss(last_hidden_state, triple_ids, is_predicted_tail)
            return {
                "align_loss": align_loss,
                "kge_loss": kge_loss,
                "struct_loss": struct_loss
            }
