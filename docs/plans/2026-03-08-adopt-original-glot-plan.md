# Adopt Original GLOT Code — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current GLOT reimplementation with the original code from ipsitmantri/GLOT, extracted into the existing modular structure.

**Architecture:** The original code uses: (1) edge-weighted token graphs, (2) no input projection — first GATConv takes raw d-dim input, JK cat output = d + K×p, (3) 4 GNN types (GAT/GCN/GIN/GINE), (4) scaled scorer hidden dim. We extract these into graph_construction.py, token_gnn.py, readout.py, and absorb glot_pooler.py into model.py.

**Tech Stack:** PyTorch, PyTorch Geometric (GATConv, GCNConv, GINConv, GINEConv, JumpingKnowledge), HuggingFace Transformers/Datasets

**Key formula change:** JK cat output_dim = input_dim + num_layers × hidden_dim (was hidden_dim × (num_layers + 1))

---

### Task 1: Graph Construction — Add Edge Weights

**Files:**
- Modify: `glot/graph_construction.py`
- Modify: `tests/test_graph_construction.py`

**Step 1: Update tests for edge_attr**

Add tests for edge weights to `tests/test_graph_construction.py`. Keep all existing tests (they should still pass). Add:

```python
def test_edge_attr_present(self):
    """Graph edges should have cosine similarity as edge_attr."""
    hidden = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.long)
    batch = build_token_graph(hidden, mask, threshold=0.0)
    assert batch.edge_attr is not None
    assert batch.edge_attr.shape == (batch.edge_index.shape[1], 1)

def test_edge_attr_values_are_similarities(self):
    """edge_attr values should be cosine similarities > threshold."""
    hidden = torch.randn(1, 4, 16)
    mask = torch.ones(1, 4, dtype=torch.long)
    threshold = 0.3
    batch = build_token_graph(hidden, mask, threshold=threshold)
    if batch.edge_attr.numel() > 0:
        assert (batch.edge_attr > threshold).all()
        assert (batch.edge_attr <= 1.0).all()

def test_edge_attr_matches_edge_count(self):
    """Number of edge_attr values equals number of edges."""
    hidden = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.long)
    batch = build_token_graph(hidden, mask, threshold=0.5)
    assert batch.edge_attr.shape[0] == batch.edge_index.shape[1]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_construction.py -v`
Expected: 3 new tests FAIL (edge_attr is None)

**Step 3: Update graph_construction.py to include edge_attr**

Replace `glot/graph_construction.py` with:

```python
import torch
from torch_geometric.data import Data, Batch


def build_token_graph(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    threshold: float = 0.6,
) -> Batch:
    """Construct edge-weighted token-similarity graphs from hidden states.

    Args:
        hidden_states: (B, L, d) token hidden states from frozen LLM.
        attention_mask: (B, L) with 1 for valid tokens, 0 for padding.
        threshold: cosine similarity threshold for edge creation.

    Returns:
        PyG Batch containing B graphs with edge_attr (cosine similarity).
    """
    graphs = []
    for i in range(hidden_states.size(0)):
        mask = attention_mask[i].bool()
        h = hidden_states[i][mask]  # (L', d)

        # Pairwise cosine similarity
        h_norm = torch.nn.functional.normalize(h, p=2, dim=-1)
        sim = h_norm @ h_norm.T  # (L', L')

        # Threshold to binary adjacency, remove self-loops
        adj = (sim > threshold).long()
        adj.fill_diagonal_(0)

        # COO edge list
        edge_index = adj.nonzero(as_tuple=False).T.contiguous()  # (2, |E|)

        # Edge weights = cosine similarity for surviving edges
        if edge_index.shape[1] > 0:
            src, dst = edge_index
            edge_attr = sim[src, dst].unsqueeze(-1)  # (|E|, 1)
        else:
            edge_attr = torch.zeros(0, 1, device=h.device)

        graphs.append(Data(x=h, edge_index=edge_index, edge_attr=edge_attr))

    return Batch.from_data_list(graphs)
```

**Step 4: Run all graph construction tests**

Run: `pytest tests/test_graph_construction.py -v`
Expected: All 11 tests PASS (8 existing + 3 new)

**Step 5: Commit**

```bash
git add glot/graph_construction.py tests/test_graph_construction.py
git commit -m "feat: add edge weights (cosine similarity) to token graphs"
```

---

### Task 2: TokenGNN — Original JK Formula + Multi-GNN

**Files:**
- Modify: `glot/token_gnn.py`
- Modify: `tests/test_token_gnn.py`

The original code does NOT have an input projection. The first GATConv takes raw d-dim input, outputs hidden_dim. JK cat concatenates: [input (d), layer1 (p), layer2 (p), ...] → d + K×p.

**Step 1: Rewrite tests for new output dims and GNN types**

Replace `tests/test_token_gnn.py` with:

```python
import torch
import pytest
from glot.token_gnn import TokenGNN


class TestTokenGNN:
    def test_output_shape_jk_cat_bert(self):
        """With BERT (d=768), K=2, p=128, JK=cat: output = 768 + 2*128 = 1024."""
        gnn = TokenGNN(input_dim=768, hidden_dim=128, num_layers=2, jk_mode="cat")
        x = torch.randn(10, 768)
        edge_index = torch.tensor(
            [list(range(9)) + list(range(1, 10)),
             list(range(1, 10)) + list(range(9))],
            dtype=torch.long,
        )
        out = gnn(x, edge_index)
        assert out.shape == (10, 1024)

    def test_output_dim_attribute(self):
        """output_dim attribute matches d + K*p."""
        gnn = TokenGNN(input_dim=32, hidden_dim=64, num_layers=2, jk_mode="cat")
        assert gnn.output_dim == 32 + 2 * 64  # 160

    def test_output_dim_4_layers(self):
        """With K=4, p=256: output = d + 4*256."""
        gnn = TokenGNN(input_dim=768, hidden_dim=256, num_layers=4, jk_mode="cat")
        x = torch.randn(5, 768)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (5, 768 + 4 * 256)  # 1792
        assert gnn.output_dim == 1792

    def test_no_edges_still_works(self):
        """GNN should handle isolated nodes (no edges) gracefully."""
        gnn = TokenGNN(input_dim=16, hidden_dim=8, num_layers=2, jk_mode="cat")
        x = torch.randn(5, 16)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (5, 16 + 2 * 8)  # 32

    def test_gradients_flow(self):
        """Gradients should flow through all parameters."""
        gnn = TokenGNN(input_dim=16, hidden_dim=8, num_layers=2, jk_mode="cat")
        x = torch.randn(5, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = gnn(x, edge_index)
        loss = out.sum()
        loss.backward()
        for name, param in gnn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_different_input_dims(self):
        """Should work with various LLM hidden dimensions."""
        for input_dim in [768, 2048, 4096]:
            gnn = TokenGNN(input_dim=input_dim, hidden_dim=128, num_layers=2, jk_mode="cat")
            x = torch.randn(3, input_dim)
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            out = gnn(x, edge_index)
            assert out.shape == (3, input_dim + 2 * 128)

    def test_gnn_type_gcn(self):
        """GCN backend should work."""
        gnn = TokenGNN(input_dim=32, hidden_dim=16, num_layers=2, jk_mode="cat", gnn_type="GCN")
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (5, 32 + 2 * 16)

    def test_gnn_type_gin(self):
        """GIN backend should work."""
        gnn = TokenGNN(input_dim=32, hidden_dim=16, num_layers=2, jk_mode="cat", gnn_type="GIN")
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (5, 32 + 2 * 16)

    def test_gnn_type_gine(self):
        """GINE backend should work with edge_attr."""
        gnn = TokenGNN(input_dim=32, hidden_dim=16, num_layers=2, jk_mode="cat", gnn_type="GINE")
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(4, 1)
        out = gnn(x, edge_index, edge_attr=edge_attr)
        assert out.shape == (5, 32 + 2 * 16)

    def test_edge_attr_passed_to_gat(self):
        """GAT should accept edge_attr without error."""
        gnn = TokenGNN(input_dim=16, hidden_dim=8, num_layers=2, jk_mode="cat", gnn_type="GAT")
        x = torch.randn(5, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(3, 1)
        out = gnn(x, edge_index, edge_attr=edge_attr)
        assert out.shape == (5, 16 + 2 * 8)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_token_gnn.py -v`
Expected: Most tests FAIL (wrong output dims, no gnn_type param, no edge_attr)

**Step 3: Rewrite token_gnn.py with original architecture**

Replace `glot/token_gnn.py` with:

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv, JumpingKnowledge


class TokenGNN(nn.Module):
    """GNN that refines token representations via message passing.

    Original architecture: no input projection, first GNN layer takes raw
    d-dim input. JK cat output = input_dim + num_layers × hidden_dim.
    Supports GAT, GCN, GIN, GINE backends.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        jk_mode: str = "cat",
        gnn_type: str = "GAT",
    ):
        super().__init__()
        self.gnn_type = gnn_type

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            self.convs.append(self._make_conv(gnn_type, in_channels, hidden_dim))

        self.activation = nn.ReLU()
        self.jk = JumpingKnowledge(mode=jk_mode)

        if jk_mode == "cat":
            self.output_dim = input_dim + num_layers * hidden_dim
        else:
            self.output_dim = hidden_dim

    @staticmethod
    def _make_conv(gnn_type: str, in_channels: int, out_channels: int) -> nn.Module:
        if gnn_type == "GAT":
            return GATConv(in_channels, out_channels, edge_dim=1)
        elif gnn_type == "GCN":
            return GCNConv(in_channels, out_channels)
        elif gnn_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            return GINConv(mlp)
        elif gnn_type == "GINE":
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            return GINEConv(mlp, edge_dim=1)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer_outputs = [x]  # Include raw input (dim d) in JK
        h = x
        for conv in self.convs:
            if self.gnn_type in ("GAT", "GINE") and edge_attr is not None:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            h = self.activation(h)
            layer_outputs.append(h)

        return self.jk(layer_outputs)
```

**Step 4: Run tests**

Run: `pytest tests/test_token_gnn.py -v`
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add glot/token_gnn.py tests/test_token_gnn.py
git commit -m "feat: rewrite TokenGNN with original JK formula and multi-GNN support"
```

---

### Task 3: Readout — Scaled Scorer Hidden Dimension

**Files:**
- Modify: `glot/readout.py`
- Modify: `tests/test_readout.py`

**Step 1: Add test for scaled scorer**

Add to `tests/test_readout.py`:

```python
def test_scorer_hidden_scales_with_dim(self):
    """Scorer hidden dim should be max(128, input_dim // 2)."""
    readout = AttentionReadout(input_dim=1024)
    # hidden should be max(128, 1024//2) = 512
    first_linear = readout.attention[0]
    assert first_linear.out_features == 512

def test_scorer_hidden_minimum_128(self):
    """Scorer hidden dim should be at least 128."""
    readout = AttentionReadout(input_dim=64)
    first_linear = readout.attention[0]
    assert first_linear.out_features == 128
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_readout.py::TestAttentionReadout::test_scorer_hidden_scales_with_dim -v`
Expected: FAIL (current uses input_dim as hidden, not max(128, input_dim//2))

**Step 3: Update readout.py**

Replace `glot/readout.py` with:

```python
import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.nn import global_add_pool


class AttentionReadout(nn.Module):
    """Learned attention-weighted aggregation over graph nodes.

    Computes a scalar importance score per token via an MLP,
    applies per-graph softmax, then weighted-sums to produce
    one vector per sentence.

    Scorer hidden dim scales with input: max(128, input_dim // 2).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        scorer_hidden = max(128, input_dim // 2)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, scorer_hidden),
            nn.Tanh(),
            nn.Linear(scorer_hidden, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)  # (N_total,)
        weights = pyg_softmax(scores, batch)  # (N_total,)
        weighted = x * weights.unsqueeze(-1)  # (N_total, D)
        return global_add_pool(weighted, batch)  # (B, D)
```

**Step 4: Run tests**

Run: `pytest tests/test_readout.py -v`
Expected: All 7 tests PASS (5 existing + 2 new). Note: `test_single_node_graph` may need updating since the attention layer dimensions changed — the single-node case still produces (1, 8) output, but the value won't be exactly equal to x anymore since scorer hidden != input_dim. Update that test:

In `test_single_node_graph`, change assertion to just check shape (the exact value depends on weights, not guaranteed to equal x with different hidden dim):

```python
def test_single_node_graph(self):
    """Graph with one node should return that node's representation (scaled by softmax=1)."""
    readout = AttentionReadout(input_dim=8)
    x = torch.randn(1, 8)
    batch = torch.tensor([0])
    z = readout(x, batch)
    assert z.shape == (1, 8)
    # With single node, softmax weight = 1.0, so z = 1.0 * x = x
    assert torch.allclose(z, x, atol=1e-6)
```

This test should still pass — single node softmax is always 1.0 regardless of scorer architecture.

**Step 5: Commit**

```bash
git add glot/readout.py tests/test_readout.py
git commit -m "feat: scale readout scorer hidden dim with max(128, input_dim // 2)"
```

---

### Task 4: Absorb GLOTPooler into model.py

**Files:**
- Modify: `glot/model.py`
- Delete: `glot/glot_pooler.py`
- Modify: `glot/__init__.py`
- Modify: `tests/test_model.py`
- Delete: `tests/test_glot_pooler.py`

**Step 1: Merge GLOTPooler tests into test_model.py**

Add GLOTPooler tests to `tests/test_model.py` (ported from test_glot_pooler.py with updated dims):

```python
class TestGLOTPooler:
    def test_output_shape_default_config(self):
        """Default config: d=768, hidden=128, layers=2, jk=cat -> 768 + 2*128 = 1024."""
        pooler = GLOTPooler(input_dim=768)
        hidden = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 1024)

    def test_output_dim_attribute(self):
        """output_dim = input_dim + num_layers * hidden_dim."""
        pooler = GLOTPooler(input_dim=768, hidden_dim=128, num_gnn_layers=2, jk_mode="cat")
        assert pooler.output_dim == 768 + 2 * 128

    def test_handles_padding(self):
        """Padding tokens should not affect the output shape."""
        pooler = GLOTPooler(input_dim=32, hidden_dim=16, num_gnn_layers=2)
        hidden = torch.randn(2, 8, 32)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0]])
        z = pooler(hidden, mask)
        assert z.shape == (2, 32 + 2 * 16)  # 64

    def test_single_token_sentence(self):
        """A sentence with only 1 valid token should still produce output."""
        pooler = GLOTPooler(input_dim=16, hidden_dim=8, num_gnn_layers=2)
        hidden = torch.randn(1, 5, 16)
        mask = torch.tensor([[1, 0, 0, 0, 0]])
        z = pooler(hidden, mask)
        assert z.shape == (1, 16 + 2 * 8)  # 32

    def test_gradients_flow_through_pooler(self):
        """Gradients should flow through the entire pooler pipeline."""
        pooler = GLOTPooler(input_dim=16, hidden_dim=8, num_gnn_layers=2)
        hidden = torch.randn(2, 5, 16)
        mask = torch.ones(2, 5, dtype=torch.long)
        z = pooler(hidden, mask)
        loss = z.sum()
        loss.backward()
        for name, param in pooler.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_threshold_parameter_used(self):
        """Different thresholds should produce different outputs."""
        torch.manual_seed(42)
        hidden = torch.randn(1, 10, 32)
        mask = torch.ones(1, 10, dtype=torch.long)

        pooler_low = GLOTPooler(input_dim=32, hidden_dim=16, num_gnn_layers=2, threshold=0.1)
        pooler_high = GLOTPooler(input_dim=32, hidden_dim=16, num_gnn_layers=2, threshold=0.9)

        pooler_high.load_state_dict(pooler_low.state_dict())

        z_low = pooler_low(hidden, mask)
        z_high = pooler_high(hidden, mask)
        assert not torch.allclose(z_low, z_high, atol=1e-4)

    def test_gnn_type_parameter(self):
        """gnn_type parameter should be passed to TokenGNN."""
        for gnn_type in ["GAT", "GCN", "GIN"]:
            pooler = GLOTPooler(input_dim=32, hidden_dim=16, num_gnn_layers=2, gnn_type=gnn_type)
            hidden = torch.randn(2, 5, 32)
            mask = torch.ones(2, 5, dtype=torch.long)
            z = pooler(hidden, mask)
            assert z.shape == (2, 32 + 2 * 16)
```

Also update the import in test_model.py:

```python
from glot.model import create_pooler_and_head, GLOTPooler
```

And update TestCreatePoolerAndHead tests for new dims — for example `test_glot_classification` and `test_pair_classification_head` should still work since they don't assert on intermediate dims, just final logits shape.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: New TestGLOTPooler tests FAIL (GLOTPooler not in model.py yet)

**Step 3: Move GLOTPooler into model.py**

Replace `glot/model.py` with:

```python
import torch.nn as nn
from glot.graph_construction import build_token_graph
from glot.token_gnn import TokenGNN
from glot.readout import AttentionReadout
from glot.baselines import MeanPooler, MaxPooler, CLSPooler, AdaPool, EOSPooler


class GLOTPooler(nn.Module):
    """GLOT pooling module: graph construction + GNN + readout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        jk_mode: str = "cat",
        threshold: float = 0.6,
        gnn_type: str = "GAT",
    ):
        super().__init__()
        self.threshold = threshold
        self.token_gnn = TokenGNN(
            input_dim, hidden_dim, num_gnn_layers, jk_mode, gnn_type=gnn_type,
        )
        self.readout = AttentionReadout(self.token_gnn.output_dim)
        self.output_dim = self.token_gnn.output_dim

    def forward(self, hidden_states, attention_mask):
        batch_data = build_token_graph(hidden_states, attention_mask, self.threshold)
        refined = self.token_gnn(
            batch_data.x, batch_data.edge_index, edge_attr=batch_data.edge_attr,
        )
        return self.readout(refined, batch_data.batch)


def create_pooler_and_head(
    pooler_type: str,
    input_dim: int,
    num_classes: int,
    task_type: str = "classification",
    glot_config: dict | None = None,
) -> tuple[nn.Module, nn.Module]:
    """Create a pooler and task head.

    Returns:
        (pooler, head) tuple. The pooler maps (B, L, d) -> (B, D).
        The head maps (B, D) -> (B, num_classes) for classification,
        or (B, 2*D) -> (B, num_classes) for pair classification.
    """
    if pooler_type == "glot":
        pooler = GLOTPooler(input_dim=input_dim, **(glot_config or {}))
        pool_dim = pooler.output_dim
    elif pooler_type == "mean":
        pooler = MeanPooler()
        pool_dim = input_dim
    elif pooler_type == "max":
        pooler = MaxPooler()
        pool_dim = input_dim
    elif pooler_type == "cls":
        pooler = CLSPooler(is_decoder=False)
        pool_dim = input_dim
    elif pooler_type == "eos":
        pooler = EOSPooler()
        pool_dim = input_dim
    elif pooler_type == "adapool":
        pooler = AdaPool(input_dim)
        pool_dim = input_dim
    else:
        raise ValueError(f"Unknown pooler type: {pooler_type}")

    if task_type == "pair_classification":
        head = nn.Linear(pool_dim * 2, num_classes)
    else:
        head = nn.Linear(pool_dim, num_classes)

    return pooler, head
```

**Step 4: Update __init__.py**

Replace `glot/__init__.py` with:

```python
from glot.graph_construction import build_token_graph
from glot.token_gnn import TokenGNN
from glot.readout import AttentionReadout
from glot.model import GLOTPooler, create_pooler_and_head
from glot.baselines import MeanPooler, MaxPooler, CLSPooler, AdaPool, EOSPooler
from glot.utils import compute_metrics, load_config, GLUE_TASKS
from glot.backbone import BACKBONE_REGISTRY, get_backbone_config, load_backbone
```

**Step 5: Delete glot_pooler.py and test_glot_pooler.py**

```bash
rm glot/glot_pooler.py tests/test_glot_pooler.py
```

**Step 6: Run all tests**

Run: `pytest tests/test_model.py tests/test_graph_construction.py tests/test_token_gnn.py tests/test_readout.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add glot/model.py glot/__init__.py tests/test_model.py
git rm glot/glot_pooler.py tests/test_glot_pooler.py
git commit -m "refactor: absorb GLOTPooler into model.py, add gnn_type support"
```

---

### Task 5: Update E2E Tests and train.py for New Dims

**Files:**
- Modify: `tests/test_e2e.py`
- Modify: `train.py`

**Step 1: Update test_e2e.py for new output dims**

The e2e tests pass `glot_config={"hidden_dim": 16, "num_gnn_layers": 1}` and `glot_config={"hidden_dim": 8, "num_gnn_layers": 1}`. These should still work since create_pooler_and_head handles dims internally. But verify.

Run: `pytest tests/test_e2e.py -v`

If tests fail, update glot_config dicts to include `"gnn_type": "GAT"` if needed. The factory should default gnn_type to "GAT" so this should pass without changes.

**Step 2: Update train.py to pass gnn_type from config**

In `train.py`, update the glot_config dict (around line 103-108):

```python
    glot_config = {
        "hidden_dim": cfg["glot"]["hidden_dim"],
        "num_gnn_layers": cfg["glot"]["num_layers"],
        "jk_mode": cfg["glot"]["jk_mode"],
        "threshold": cfg["glot"]["threshold"],
        "gnn_type": cfg["glot"].get("gnn_type", "GAT"),
    }
```

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (except test_glot_pooler.py which was deleted)

**Step 4: Commit**

```bash
git add train.py tests/test_e2e.py
git commit -m "feat: pass gnn_type through config, verify e2e tests"
```

---

### Task 6: Update Config

**Files:**
- Modify: `configs/default.yaml`
- Modify: `glot/utils.py` (if config loading needs updates)

**Step 1: Update default.yaml**

```yaml
backbone:
  name: "bert-base-uncased"
  freeze: true

glot:
  gnn_type: "GAT"
  hidden_dim: 128
  num_layers: 2
  jk_mode: "cat"
  threshold: 0.6

training:
  epochs: 2
  lr: 0.0002
  weight_decay: 0.0
  batch_size: 32
  eval_batch_size: 64
  seed: 42

task:
  name: "sst2"
  type: "classification"
  num_classes: 2
  max_length: 128
  metric: "accuracy"
```

Config already has `gnn_type: "GAT"`. No change needed for the YAML. Just verify train.py reads it.

**Step 2: Run config-related tests**

Run: `pytest tests/test_utils.py -v`
Expected: All PASS

**Step 3: Commit (only if changes were needed)**

```bash
git add configs/default.yaml
git commit -m "chore: verify config supports gnn_type"
```

---

### Task 7: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update architecture section**

Update CLAUDE.md to reflect:
- JK output dim formula: `input_dim + num_layers × hidden_dim`
- 4 GNN types: GAT, GCN, GIN, GINE
- Edge-weighted graphs (cosine similarity as edge_attr)
- Scaled scorer: `max(128, output_dim // 2)`
- GLOTPooler now lives in `model.py` (not separate file)
- Remove `glot_pooler.py` from project structure listing

Key sections to update:
- **Project Structure** — remove glot_pooler.py, add note about model.py containing GLOTPooler
- **Architecture (3-Stage Pipeline)** — update output dim formula, mention edge weights
- **Key Module APIs** — update TokenGNN signature (add gnn_type), update GLOTPooler location
- **Coding Conventions** — add gnn_type convention

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for original GLOT architecture"
```

---

### Task 8: Full Test Suite Validation

**Step 1: Run complete test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS. No test file should import from `glot.glot_pooler`.

**Step 2: Verify no stale imports**

Run: `grep -r "from glot.glot_pooler" .` (excluding .git)
Expected: No matches

Run: `grep -r "glot_pooler" tests/`
Expected: No matches

**Step 3: Verify param count is reasonable**

Add a quick check (can be run ad-hoc, not committed):
```python
from glot.model import GLOTPooler
pooler = GLOTPooler(input_dim=768)
total = sum(p.numel() for p in pooler.parameters())
print(f"GLOT params (BERT): {total:,}")
# Expected: ~8-9M trainable params
```

---

### Summary of Changes

| File | Action | Key Change |
|------|--------|------------|
| `glot/graph_construction.py` | Modified | Added `edge_attr` (cosine sim) to Data objects |
| `glot/token_gnn.py` | Rewritten | No input_proj, first conv takes d-dim, JK = d + K×p, 4 GNN types, edge_attr support |
| `glot/readout.py` | Modified | Scorer hidden = `max(128, input_dim // 2)` |
| `glot/model.py` | Expanded | Absorbed GLOTPooler, passes edge_attr and gnn_type |
| `glot/glot_pooler.py` | Deleted | Merged into model.py |
| `glot/__init__.py` | Updated | Import GLOTPooler from model |
| `train.py` | Updated | Pass gnn_type from config |
| `configs/default.yaml` | Verified | Already has gnn_type |
| `CLAUDE.md` | Updated | New architecture docs |
| `tests/test_graph_construction.py` | Updated | +3 edge_attr tests |
| `tests/test_token_gnn.py` | Rewritten | New dims, 4 GNN type tests, edge_attr tests |
| `tests/test_readout.py` | Updated | +2 scaled scorer tests |
| `tests/test_model.py` | Expanded | +7 GLOTPooler tests (from deleted test_glot_pooler.py) |
| `tests/test_glot_pooler.py` | Deleted | Merged into test_model.py |
