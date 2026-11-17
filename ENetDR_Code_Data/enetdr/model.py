import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_add_pool
from .evidence import EntityEvidenceUpdate

class LLM_GNN_Model(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256,
                 num_doc_cls=6, num_pat_cls=6,
                 type_dim=16, heads=4, use_token_evidence=True, use_retrieval_gate=True,
                 edge_chunk_size=128, legacy_graph=False):
        super().__init__()

        self.utt_proj = nn.Linear(input_dim + 4, hidden_dim)
        self.ent_proj = nn.Linear(input_dim + 4, hidden_dim) 

        self.node_type_emb = nn.Embedding(3, type_dim)

        self.edge_type_emb = nn.Embedding(5, type_dim)

        gnn_in_dim = hidden_dim + type_dim

        self.conv1 = TransformerConv(gnn_in_dim, hidden_dim, heads=heads, concat=True, dropout=0.1, edge_dim=type_dim)
        self.conv2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=0.1, edge_dim=type_dim)
        
        self.evidence_update = None if legacy_graph else EntityEvidenceUpdate(
            hidden_dim, input_dim, type_dim, use_token_evidence,
            use_retrieval_gate, edge_chunk_size)
        self.layer_norm = nn.LayerNorm(hidden_dim) 

        self.ent_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 2)
        )

        self.doc_feat_proj = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.2)
        )
        self.doc_classifier = nn.Linear(128, num_doc_cls - 1)

        self.pat_gate_net = nn.Sequential(
            nn.Linear(hidden_dim + 128, 1),
            nn.Sigmoid()
        )
        
        self.pat_feat_proj = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128), 
            nn.ReLU(), nn.Dropout(0.2)
        )
        self.pat_classifier = nn.Linear(128, num_pat_cls - 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        node_type = data.node_type.to(x.device)

        x_proj = torch.zeros(x.size(0), self.utt_proj.out_features, device=x.device)
        
        utt_mask = (node_type != 2) 
        ent_mask = (node_type == 2) 

        if utt_mask.any():
            x_proj[utt_mask] = self.utt_proj(x[utt_mask])
        if ent_mask.any():
            x_proj[ent_mask] = self.ent_proj(x[ent_mask])

        t_emb = self.node_type_emb(node_type)
        x_in = torch.cat([x_proj, t_emb], dim=-1) 

        edge_types = data.edge_attr[:, 0].long().to(x.device)
        edge_weights = data.edge_attr[:, 1].unsqueeze(1).to(x.device) 

        e_emb = self.edge_type_emb(edge_types)

        e_emb = e_emb * edge_weights 

        x = self.conv1(x_in, edge_index, edge_attr=e_emb)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=e_emb)
        x = self.layer_norm(x)

        cand_x = (x[data.candidate_mask] if self.evidence_update is None
                  else self.evidence_update(x, data, e_emb))

        ent_logits = self.ent_head(cand_x)

        doc_feat = self.doc_feat_proj(cand_x) 
        doc_state_logits = self.doc_classifier(doc_feat)

        cat_feat = torch.cat([cand_x, doc_feat], dim=-1)

        gate = self.pat_gate_net(cat_feat) 

        pat_input = torch.cat([cand_x, doc_feat * gate], dim=-1)
        
        pat_feat = self.pat_feat_proj(pat_input)
        pat_state_logits = self.pat_classifier(pat_feat)

        return ent_logits, doc_state_logits, pat_state_logits
