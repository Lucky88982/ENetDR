import torch
import torch.nn as nn
from torch_geometric.utils import softmax as graph_softmax
import math

class EntityEvidenceUpdate(nn.Module):
    def __init__(self, hidden_dim, token_dim, edge_dim, use_token=True,
                 use_retrieval=True, chunk_size=128):
        super().__init__()
        self.use_token = use_token
        self.use_retrieval = use_retrieval
        self.chunk_size = max(1, chunk_size)
        self.h = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.root = nn.Linear(hidden_dim, hidden_dim)
        if use_token:
            self.token_query = nn.Linear(hidden_dim, hidden_dim)
            self.token_key = nn.Linear(token_dim, hidden_dim)
            self.token_value = nn.Linear(token_dim, hidden_dim)
        if use_retrieval:
            self.retrieval_encoder = nn.Sequential(nn.Linear(5, hidden_dim), nn.ReLU())
            self.update_gate = nn.Linear(3 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_h, data, edge_features):
        entity_nodes = data.candidate_mask.nonzero(as_tuple=False).flatten()
        base = node_h[entity_nodes]
        if not len(entity_nodes):
            return base
        src, dst = data.edge_index
        keep = (data.node_type[src] != 2) & (data.node_type[dst] == 2)
        src, dst = src[keep], dst[keep]
        ef = edge_features[keep]
        local_id = torch.full((len(node_h),), -1, device=node_h.device, dtype=torch.long)
        local_id[entity_nodes] = torch.arange(len(entity_nodes), device=node_h.device)
        targets = local_id[dst]
        messages = torch.zeros_like(base)
        connected = torch.zeros(len(base), dtype=torch.bool, device=base.device)
        if len(src):
            if self.use_token:
                tokens = data.token_states.to(dtype=node_h.dtype)
                owner, pos = data.token_owner, data.token_pos
                
                length = int(pos.max().item()) + 1
                keys = node_h.new_zeros((len(node_h), length, self.h))
                vals = torch.zeros_like(keys)
                valid = torch.zeros((len(node_h), length), dtype=torch.bool, device=node_h.device)
                keys[owner, pos] = self.token_key(tokens)
                vals[owner, pos] = self.token_value(tokens)
                valid[owner, pos] = True
                chunks = []
                for start in range(0, len(src), self.chunk_size):
                    ss, dd = src[start:start+self.chunk_size], dst[start:start+self.chunk_size]
                    q = self.token_query(node_h[dd])
                    scores = (keys[ss] * q[:, None, :]).sum(-1) / math.sqrt(self.h)
                    scores = scores.masked_fill(~valid[ss], float('-inf'))
                    weights = scores.softmax(-1)
                    chunks.append((weights[:, :, None] * vals[ss]).sum(1))
                evidence = torch.cat(chunks)
            else:
                evidence = node_h[src]
            edge = self.edge(ef)
            key, value = self.key(evidence) + edge, self.value(evidence) + edge
            logits = (self.query(node_h[dst]) * key).sum(-1) / math.sqrt(self.h)
            alpha = graph_softmax(logits, targets, num_nodes=len(base))
            messages.index_add_(0, targets, alpha[:, None] * value)
            connected[targets] = True
        root = self.root(base)
        if self.use_retrieval:
            cond = self.retrieval_encoder(data.retrieval_features.to(base.dtype))
            gate = torch.sigmoid(self.update_gate(torch.cat([base, messages, cond], -1)))
            
            gate = gate * connected[:, None].to(gate.dtype)
            updated = gate * messages + (1.0 - gate) * root
        else:
            updated = root + messages
        return self.norm(updated)
