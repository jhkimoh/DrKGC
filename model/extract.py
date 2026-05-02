import torch
from torch import nn
import torch.nn.functional as F
import math

class KG_extract(nn.Module):
    def __init__(self, hidden_dim, E_dim, R_dim, batch_size, tau=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.E_dim = E_dim
        self.R_dim = R_dim
        self.batch_size = batch_size
        self.W_enc_h = nn.Linear(self.hidden_dim, self.E_dim)
        self.W_enc_r = nn.Linear(self.hidden_dim, self.R_dim)
        self.W_enc_t = nn.Linear(self.hidden_dim, self.E_dim)
        self.W_dec_h = nn.Linear(self.E_dim, self.hidden_dim)
        self.W_dec_r = nn.Linear(self.R_dim, self.hidden_dim)
        self.W_dec_t = nn.Linear(self.E_dim, self.hidden_dim)
        init_val = 1.0 / tau
        inv_softplus_init = math.log(math.exp(init_val) - 1.0) if init_val < 20.0 else init_val
        self.inv_t_param = nn.Parameter(torch.tensor(inv_softplus_init))
        self.register_buffer("dead_h", torch.zeros(self.E_dim))
        self.register_buffer("dead_r", torch.zeros(self.R_dim))
        self.register_buffer("dead_t", torch.zeros(self.E_dim))

    def cal_reconstruction_loss(self, x_reconstruct, x):
        recon_loss = F.mse_loss(x_reconstruct, x)
        return recon_loss
    
    def cal_label_loss(self, h_logits, r_logits, t_logits, entity_ids, triple_ids, is_predicted_tail):
        h_target = F.one_hot(triple_ids[:, 0], num_classes=self.E_dim).float()
        r_target = F.one_hot(triple_ids[:, 1], num_classes=self.R_dim).float()
        t_target = F.one_hot(triple_ids[:, 2], num_classes=self.E_dim).float()
        cand_target = torch.zeros_like(h_target).scatter_(1, entity_ids, 1.0)
        is_tail = is_predicted_tail.unsqueeze(1)
        final_h_target = torch.where(is_tail, h_target, cand_target)
        final_t_target = torch.where(is_tail, cand_target, t_target)
        loss_bce = (
            F.binary_cross_entropy_with_logits(h_logits, final_h_target) +
            F.binary_cross_entropy_with_logits(r_logits, r_target) +
            F.binary_cross_entropy_with_logits(t_logits, final_t_target)
        )
        return loss_bce

    def cal_kgc_loss(self, entity_ids, triple_ids, is_predicted_tail):
        # 1. Bias가 배제된 순수 임베딩 추출 (W_dec의 전치 행렬)
        # weight 형태: [hidden_dim, E_dim] -> .t() -> [E_dim, hidden_dim]
        D_H = self.W_dec_h.weight.t()
        D_R = self.W_dec_r.weight.t()
        D_T = self.W_dec_t.weight.t()
        # 2. 정답(Ground Truth)의 고정된 임베딩 가져오기
        h_fixed = D_H[triple_ids[:, 0]] # [batch_size, hidden_dim]
        r_fixed = D_R[triple_ids[:, 1]]
        t_fixed = D_T[triple_ids[:, 2]]
        # 3. 주어진 20개의 후보군(Candidates) # entity_ids: [batch_size, num_candidates]
        cand_H = D_H[entity_ids] # [batch_size, num_candidates, hidden_dim]
        cand_T = D_T[entity_ids]
        # Tail을 예측할 때 (h + r 이 기준점)
        q_tail = (h_fixed + r_fixed).unsqueeze(1) # [batch_size, 1, hidden_dim]
        d_tail = torch.norm(q_tail - cand_T, p=2, dim=2) # [batch_size, num_candidates]
        # Head를 예측할 때 (t - r 이 기준점)
        q_head = (t_fixed - r_fixed).unsqueeze(1)
        d_head = torch.norm(cand_H - q_head, p=2, dim=2)
        # 5. 라우팅 (is_predicted_tail에 따라 거리 선택)
        is_tail = is_predicted_tail.unsqueeze(1)
        distances = torch.where(is_tail, d_tail, d_head) # [batch_size, num_candidates]
        # 6. 정답(Target) 라벨링 # 현재 예측해야 할 진짜 ID가 무엇인지 찾기
        target_entity = torch.where(is_predicted_tail, triple_ids[:, 2], triple_ids[:, 0])
        matches = (entity_ids == target_entity.unsqueeze(1)) #(B,20)==(B,1)
        valid_mask = matches.any(dim=1) #(B)
        labels = torch.argmax(matches.long(), dim=1)
        inv_t = torch.clamp(F.softplus(self.inv_t_param), max=100.0)
        logits = -distances * inv_t
        loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
        final_loss = (loss_per_sample * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-8)
        return final_loss

    def forward(self, x, query_ids, entity_ids, triple_ids, is_predicted_tail):
        #인코더 통과
        h_logits, r_logits, t_logits = self.W_enc_h(x), self.W_enc_r(x), self.W_enc_t(x)
        h_acts, r_acts, t_acts = F.relu(h_logits), F.relu(r_logits), F.relu(t_logits)
        #디코더 통과
        h_recon, r_recon, t_recon = self.W_dec_h(h_acts), self.W_dec_r(r_acts), self.W_dec_t(t_acts)
        with torch.no_grad(): # 이건 학습되는 가중치가 아니므로 그래디언트 계산을 끕니다.
            # 배치(Batch) 내에서 0보다 큰(활성화된) 횟수를 더해줍니다.
            # h_acts 형태: [batch_size, E_dim] -> sum(dim=0) -> [E_dim]
            self.dead_h += (h_acts > 0).sum(dim=0)
            self.dead_r += (r_acts > 0).sum(dim=0)
            self.dead_t += (t_acts > 0).sum(dim=0)
        #reconstruction_loss
        x_reconstruct = h_recon + r_recon + t_recon
        reconstruction_loss = self.cal_reconstruction_loss(x_reconstruct, x)
        #label_loss
        label_loss = self.cal_label_loss(h_logits, r_logits, t_logits, entity_ids, triple_ids, is_predicted_tail)
        #kgc_loss
        kgc_loss = self.cal_kgc_loss(entity_ids, triple_ids, is_predicted_tail)
        loss = reconstruction_loss + label_loss + kgc_loss
        return loss