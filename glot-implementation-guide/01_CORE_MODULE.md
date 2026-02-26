# 01: Core GLOT Module Implementation

## Overview
This file details how to implement the three core components of GLOT:
1. Token Graph Construction
2. TOKEN-GNN (message passing)
3. Readout Layer

## Dependencies
```
torch>=2.0
torch-geometric>=2.4
transformers>=4.36
```

---

## Step 1: Token Graph Construction (`graph_construction.py`)

### Purpose
Given token hidden states `X ∈ R^{L×d}` from a frozen LLM, construct a sparse token-similarity graph `G = (V, E)`.

### Algorithm
```
Input: X ∈ R^{L×d} (token hidden states), mask ∈ {0,1}^L (attention mask), τ (threshold)
Output: edge_index ∈ R^{2×|E|} (COO format sparse edge list)

1. Filter X to only valid (non-padding) tokens using mask
2. Compute pairwise cosine similarity matrix S ∈ R^{L'×L'} where L' = sum(mask)
   S_ij = (x_i · x_j) / (||x_i|| · ||x_j||)
3. Create adjacency: A_ij = 1 if S_ij > τ, else 0
4. Remove self-loops (set diagonal of A to 0) — OPTIONAL: paper doesn't explicitly state this
5. Convert A to edge_index in COO format (2 × |E| tensor)
```

### Implementation Notes
- Use `torch.nn.functional.cosine_similarity` or manual computation for batched operation
- For batched processing, use PyTorch Geometric's `Batch` to combine individual graphs
- The threshold τ is a hyperparameter (default: 0.6, search space: {0.1, 0.3, 0.6})
- The graph construction is done **per sentence** — each sentence gets its own graph
- Complexity: O(L²) for edge formation, but negligible compared to LLM forward pass (see Table 10)

### Key Code Pattern
```python
import torch
from torch_geometric.data import Data, Batch

def build_token_graph(hidden_states: torch.Tensor, attention_mask: torch.Tensor, threshold: float = 0.6):
    """
    Args:
        hidden_states: (B, L, d) - token hidden states from frozen LLM
        attention_mask: (B, L) - 1 for valid tokens, 0 for padding
        threshold: cosine similarity threshold for edge creation
    Returns:
        batch: PyG Batch object containing B graphs
    """
    graphs = []
    for i in range(hidden_states.size(0)):
        mask = attention_mask[i].bool()
        h = hidden_states[i][mask]  # (L', d) valid tokens only
        
        # Normalize for cosine similarity
        h_norm = torch.nn.functional.normalize(h, p=2, dim=-1)
        sim = h_norm @ h_norm.T  # (L', L')
        
        # Threshold to create adjacency
        adj = (sim > threshold).long()
        adj.fill_diagonal_(0)  # remove self-loops (optional)
        
        # Convert to edge_index
        edge_index = adj.nonzero(as_tuple=False).T  # (2, |E|)
        
        graphs.append(Data(x=h, edge_index=edge_index))
    
    return Batch.from_data_list(graphs)
```

### Performance Optimization
- Pre-compute and cache hidden states before training (paper mentions this in Section B.1)
- Graph construction at L=32K takes ~303ms, which is ~1.3% of total runtime (Table 10)

---

## Step 2: TOKEN-GNN (`token_gnn.py`)

### Purpose
Refine token representations by propagating information across the token graph using K layers of GNN.

### Architecture Details
- **GNN type**: GATConv (Graph Attention Network) — paper uses `GATConv` from PyTorch Geometric
- **Number of layers K**: {2, 4} (hyperparameter, default: 2)
- **Hidden dimension p**: {64, 128, 256} (default: 128)
- **Input projection**: `W_in ∈ R^{d×p}` maps from LLM hidden dim d to GNN hidden dim p
- **Activation**: ReLU
- **Jumping Knowledge**: 'cat' mode (concatenates outputs from all layers)

### Algorithm (per GNN layer ℓ)
```
For each node i:
  1. Aggregate: a_i^(ℓ) = AGGREGATE({h_j^(ℓ) : j ∈ N(i)})     # neighbors of i
  2. Update:    h_i^(ℓ+1) = σ(W^(ℓ) · CONCAT(h_i^(ℓ), a_i^(ℓ)))  # W^(ℓ) ∈ R^{p×2p}
```

### Implementation
```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, JumpingKnowledge

class TokenGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 jk_mode: str = 'cat'):
        super().__init__()
        # Input projection: d -> p
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim))
        
        self.activation = nn.ReLU()
        
        # Jumping Knowledge
        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)
        
        # Output dimension depends on JK mode
        if jk_mode == 'cat':
            self.output_dim = hidden_dim * (num_layers + 1)  # includes input layer
        else:
            self.output_dim = hidden_dim
    
    def forward(self, x, edge_index):
        # Project input
        h = self.input_proj(x)  # (N_total, p)
        
        layer_outputs = [h]
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.activation(h)
            layer_outputs.append(h)
        
        # Jumping Knowledge aggregation
        out = self.jk(layer_outputs)  # (N_total, output_dim)
        return out
```

### GNN Architecture Alternatives (Appendix D.3)
The paper tested three GNN backends:
- **GAT** (default): Best for BERT, strong overall
- **GCN**: Slightly worse than GAT but still beats all non-graph baselines
- **GIN**: Best for Mistral-7B on some tasks, most expressive

All graph variants significantly outperform set-based AdaPool, confirming the paradigm matters more than the specific architecture.

---

## Step 3: Readout Layer (`readout.py`)

### Purpose
Aggregate the set of refined token vectors U = {u_1, ..., u_L} into a single sentence vector z.

### Algorithm
```
For each token i:
  1. Score:    m_i = v^T · tanh(W_m · u_i + b_m)    # scalar importance score
  2. Normalize: π = softmax(m)                       # attention weights over tokens
  3. Aggregate: z = Σ π_i · u_i                      # weighted sum
```

### Implementation
```python
class AttentionReadout(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1, bias=False)  # v^T projection to scalar
        )
    
    def forward(self, x, batch):
        """
        Args:
            x: (N_total, D) - all token representations across batch
            batch: (N_total,) - graph assignment vector from PyG
        Returns:
            z: (B, D) - sentence-level representations
        """
        from torch_geometric.utils import softmax as pyg_softmax
        from torch_geometric.nn import global_add_pool
        
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (N_total,)
        
        # Softmax per graph (per sentence)
        weights = pyg_softmax(scores, batch)  # (N_total,)
        
        # Weighted sum
        weighted = x * weights.unsqueeze(-1)  # (N_total, D)
        z = global_add_pool(weighted, batch)  # (B, D)
        
        return z
```

---

## Combined GLOT Module (`glot_pooler.py`)

### Full Pipeline
```python
class GLOTPooler(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_gnn_layers: int = 2,
                 jk_mode: str = 'cat', threshold: float = 0.6):
        super().__init__()
        self.threshold = threshold
        self.token_gnn = TokenGNN(input_dim, hidden_dim, num_gnn_layers, jk_mode)
        self.readout = AttentionReadout(self.token_gnn.output_dim)
        self.output_dim = self.token_gnn.output_dim
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            hidden_states: (B, L, d) from frozen LLM
            attention_mask: (B, L)
        Returns:
            z: (B, output_dim) sentence embeddings
        """
        # Step 1: Build token graphs
        batch_data = build_token_graph(hidden_states, attention_mask, self.threshold)
        
        # Step 2: GNN refinement
        refined = self.token_gnn(batch_data.x, batch_data.edge_index)
        
        # Step 3: Readout
        z = self.readout(refined, batch_data.batch)
        
        return z
```

### Parameter Count
With default config (hidden_dim=128, 2 GAT layers, JK='cat'):
- Input projection: d × 128 (e.g., 4096 × 128 = 524K for Mistral)
- GAT layers: ~2 × (128 × 128 × 4) ≈ 131K (with attention heads)
- JK + Readout: ~(128 × 3) × (128 × 3) + (128 × 3) × 1 ≈ 148K
- **Total: ~8.92M** (as reported in paper)

### Properties / Special Cases
When K=0 (no GNN layers), GLOT reduces to:
- **AdaPool** if readout weights are learnable
- **Mean pooling** if all weights are 1/L
- **CLS pooling** if weight is 1 for CLS token and 0 elsewhere

This means GLOT **generalizes** all standard pooling methods.
