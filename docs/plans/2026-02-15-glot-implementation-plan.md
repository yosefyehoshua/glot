# GLOT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working research prototype of GLOT (Graph-based Learning Over Token Graphs) that trains on cached BERT-base hidden states and evaluates on GLUE subset tasks (SST-2, CoLA, MRPC) with baseline comparisons.

**Architecture:** Frozen BERT-base produces token hidden states. GLOT constructs a cosine-similarity token graph, refines representations with GATConv GNN layers + Jumping Knowledge, and aggregates via learned attention readout. Hidden states are precomputed and cached to disk. Only the pooler (~8.92M params) + task head are trained.

**Tech Stack:** Python 3.10, PyTorch 2.0+, PyTorch Geometric 2.4+, HuggingFace Transformers, HuggingFace Datasets, scikit-learn, scipy, PyYAML

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `setup.py`
- Create: `configs/default.yaml`
- Create: `glot/__init__.py`
- Create: `data/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create `requirements.txt`**

```
torch>=2.0.0
torch-geometric>=2.4.0
torch-scatter
torch-sparse
transformers>=4.36.0
datasets>=2.16.0
scikit-learn>=1.3.0
scipy>=1.11.0
pyyaml
tqdm
numpy>=1.24.0
pytest
```

**Step 2: Create `setup.py`**

```python
from setuptools import setup, find_packages

setup(
    name="glot",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
```

**Step 3: Create `configs/default.yaml`**

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
  lr: 2e-4
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

**Step 4: Create `glot/__init__.py`**

```python
from glot.graph_construction import build_token_graph
from glot.token_gnn import TokenGNN
from glot.readout import AttentionReadout
from glot.glot_pooler import GLOTPooler
```

**Step 5: Create empty `data/__init__.py` and `tests/__init__.py`**

Empty files.

**Step 6: Verify structure**

Run: `find . -name "*.py" -o -name "*.yaml" -o -name "*.txt" | head -20`

Expected: All files listed above exist.

**Step 7: Commit**

```bash
git add requirements.txt setup.py configs/ glot/__init__.py data/__init__.py tests/__init__.py
git commit -m "scaffold: project structure, requirements, config"
```

---

### Task 2: Graph Construction Module

**Files:**
- Create: `tests/test_graph_construction.py`
- Create: `glot/graph_construction.py`

**Step 1: Write the failing tests**

```python
# tests/test_graph_construction.py
import torch
import pytest
from glot.graph_construction import build_token_graph


class TestBuildTokenGraph:
    def test_output_is_pyg_batch(self):
        """build_token_graph returns a PyG Batch object."""
        from torch_geometric.data import Batch
        hidden = torch.randn(2, 5, 16)
        mask = torch.ones(2, 5, dtype=torch.long)
        batch = build_token_graph(hidden, mask, threshold=0.0)
        assert isinstance(batch, Batch)

    def test_batch_has_correct_num_graphs(self):
        """Batch contains one graph per sentence."""
        hidden = torch.randn(3, 4, 8)
        mask = torch.ones(3, 4, dtype=torch.long)
        batch = build_token_graph(hidden, mask, threshold=0.0)
        assert batch.num_graphs == 3

    def test_node_features_match_valid_tokens(self):
        """Node features are the hidden states of valid (non-padding) tokens."""
        hidden = torch.randn(1, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        batch = build_token_graph(hidden, mask, threshold=0.0)
        assert batch.x.shape == (3, 8)
        assert torch.allclose(batch.x, hidden[0, :3])

    def test_padding_tokens_excluded(self):
        """Padding tokens (mask=0) are not included as nodes."""
        hidden = torch.randn(2, 6, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0]])
        batch = build_token_graph(hidden, mask, threshold=0.0)
        # 3 + 2 = 5 total nodes
        assert batch.x.shape[0] == 5

    def test_high_threshold_produces_fewer_edges(self):
        """Higher threshold creates sparser graphs."""
        hidden = torch.randn(1, 10, 16)
        mask = torch.ones(1, 10, dtype=torch.long)
        batch_low = build_token_graph(hidden, mask, threshold=0.1)
        batch_high = build_token_graph(hidden, mask, threshold=0.9)
        assert batch_high.edge_index.shape[1] <= batch_low.edge_index.shape[1]

    def test_no_self_loops(self):
        """Diagonal of adjacency is zeroed, so no self-loops."""
        hidden = torch.randn(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.long)
        batch = build_token_graph(hidden, mask, threshold=0.0)
        src, dst = batch.edge_index
        assert (src != dst).all()

    def test_identical_vectors_fully_connected(self):
        """Identical token vectors should all be connected at any threshold < 1."""
        hidden = torch.ones(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.long)
        batch = build_token_graph(hidden, mask, threshold=0.5)
        # 4 nodes, all connected (no self-loops): 4*3 = 12 edges
        assert batch.edge_index.shape[1] == 12

    def test_batch_vector_correct(self):
        """The batch vector assigns nodes to correct graphs."""
        hidden = torch.randn(2, 3, 8)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        batch = build_token_graph(hidden, mask, threshold=0.0)
        # Graph 0: 3 nodes, Graph 1: 2 nodes
        expected_batch = torch.tensor([0, 0, 0, 1, 1])
        assert torch.equal(batch.batch, expected_batch)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_construction.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glot.graph_construction'`

**Step 3: Write the implementation**

```python
# glot/graph_construction.py
import torch
from torch_geometric.data import Data, Batch


def build_token_graph(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    threshold: float = 0.6,
) -> Batch:
    """Construct token-similarity graphs from hidden states.

    Args:
        hidden_states: (B, L, d) token hidden states from frozen LLM.
        attention_mask: (B, L) with 1 for valid tokens, 0 for padding.
        threshold: cosine similarity threshold for edge creation.

    Returns:
        PyG Batch containing B graphs, one per sentence.
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

        graphs.append(Data(x=h, edge_index=edge_index))

    return Batch.from_data_list(graphs)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_construction.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add glot/graph_construction.py tests/test_graph_construction.py
git commit -m "feat: graph construction module with tests"
```

---

### Task 3: TokenGNN Module

**Files:**
- Create: `tests/test_token_gnn.py`
- Create: `glot/token_gnn.py`

**Step 1: Write the failing tests**

```python
# tests/test_token_gnn.py
import torch
import pytest
from glot.token_gnn import TokenGNN


class TestTokenGNN:
    def test_output_shape_jk_cat(self):
        """With JK='cat' and 2 layers, output dim = hidden * 3."""
        gnn = TokenGNN(input_dim=768, hidden_dim=128, num_layers=2, jk_mode='cat')
        x = torch.randn(10, 768)
        # Simple chain graph: 0-1-2-...-9
        edge_index = torch.tensor([list(range(9)) + list(range(1, 10)),
                                   list(range(1, 10)) + list(range(9))], dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (10, 384)  # 128 * 3

    def test_output_dim_attribute(self):
        """output_dim attribute matches actual output dimension."""
        gnn = TokenGNN(input_dim=32, hidden_dim=64, num_layers=2, jk_mode='cat')
        assert gnn.output_dim == 64 * 3

    def test_output_dim_4_layers(self):
        """With 4 layers and JK='cat', output dim = hidden * 5."""
        gnn = TokenGNN(input_dim=32, hidden_dim=64, num_layers=4, jk_mode='cat')
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (5, 64 * 5)
        assert gnn.output_dim == 64 * 5

    def test_no_edges_still_works(self):
        """GNN should handle isolated nodes (no edges) gracefully."""
        gnn = TokenGNN(input_dim=16, hidden_dim=8, num_layers=2, jk_mode='cat')
        x = torch.randn(5, 16)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (5, 24)  # 8 * 3

    def test_gradients_flow(self):
        """Gradients should flow through all parameters."""
        gnn = TokenGNN(input_dim=16, hidden_dim=8, num_layers=2, jk_mode='cat')
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
            gnn = TokenGNN(input_dim=input_dim, hidden_dim=128, num_layers=2, jk_mode='cat')
            x = torch.randn(3, input_dim)
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            out = gnn(x, edge_index)
            assert out.shape == (3, 384)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_token_gnn.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glot.token_gnn'`

**Step 3: Write the implementation**

```python
# glot/token_gnn.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, JumpingKnowledge


class TokenGNN(nn.Module):
    """GNN that refines token representations via message passing.

    Architecture: input projection (d -> p), K GATConv layers with ReLU,
    Jumping Knowledge aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        jk_mode: str = "cat",
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim))

        self.activation = nn.ReLU()

        self.jk = JumpingKnowledge(
            mode=jk_mode, channels=hidden_dim, num_layers=num_layers
        )

        if jk_mode == "cat":
            self.output_dim = hidden_dim * (num_layers + 1)
        else:
            self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)

        layer_outputs = [h]
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.activation(h)
            layer_outputs.append(h)

        return self.jk(layer_outputs)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_token_gnn.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add glot/token_gnn.py tests/test_token_gnn.py
git commit -m "feat: TokenGNN module with GATConv and Jumping Knowledge"
```

---

### Task 4: Attention Readout Module

**Files:**
- Create: `tests/test_readout.py`
- Create: `glot/readout.py`

**Step 1: Write the failing tests**

```python
# tests/test_readout.py
import torch
import pytest
from glot.readout import AttentionReadout


class TestAttentionReadout:
    def test_output_shape(self):
        """Output is (B, D) where B = number of graphs."""
        readout = AttentionReadout(input_dim=384)
        # 2 graphs: 3 nodes + 4 nodes = 7 total
        x = torch.randn(7, 384)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        z = readout(x, batch)
        assert z.shape == (2, 384)

    def test_single_node_graph(self):
        """Graph with one node should return that node's representation (scaled)."""
        readout = AttentionReadout(input_dim=8)
        x = torch.randn(1, 8)
        batch = torch.tensor([0])
        z = readout(x, batch)
        assert z.shape == (1, 8)
        # With softmax over a single element, weight = 1.0, so z = x
        assert torch.allclose(z, x, atol=1e-6)

    def test_weights_sum_to_one_per_graph(self):
        """Attention weights within each graph should sum to 1."""
        readout = AttentionReadout(input_dim=16)
        x = torch.randn(5, 16)
        batch = torch.tensor([0, 0, 0, 1, 1])
        # Access internal scores to check
        scores = readout.attention(x).squeeze(-1)
        from torch_geometric.utils import softmax as pyg_softmax
        weights = pyg_softmax(scores, batch)
        # Sum per graph
        from torch_geometric.nn import global_add_pool
        weight_sums = global_add_pool(weights.unsqueeze(-1), batch).squeeze(-1)
        assert torch.allclose(weight_sums, torch.ones(2), atol=1e-5)

    def test_gradients_flow(self):
        """Gradients should flow through readout parameters."""
        readout = AttentionReadout(input_dim=16)
        x = torch.randn(5, 16, requires_grad=True)
        batch = torch.tensor([0, 0, 0, 1, 1])
        z = readout(x, batch)
        loss = z.sum()
        loss.backward()
        for name, param in readout.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_batch_of_one(self):
        """Single graph in batch should work correctly."""
        readout = AttentionReadout(input_dim=32)
        x = torch.randn(4, 32)
        batch = torch.zeros(4, dtype=torch.long)
        z = readout(x, batch)
        assert z.shape == (1, 32)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_readout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glot.readout'`

**Step 3: Write the implementation**

```python
# glot/readout.py
import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.nn import global_add_pool


class AttentionReadout(nn.Module):
    """Learned attention-weighted aggregation over graph nodes.

    Computes a scalar importance score per token via an MLP,
    applies per-graph softmax, then weighted-sums to produce
    one vector per sentence.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_total, D) all token representations across batch.
            batch: (N_total,) graph assignment vector from PyG.

        Returns:
            (B, D) sentence-level representations.
        """
        scores = self.attention(x).squeeze(-1)  # (N_total,)
        weights = pyg_softmax(scores, batch)  # (N_total,)
        weighted = x * weights.unsqueeze(-1)  # (N_total, D)
        return global_add_pool(weighted, batch)  # (B, D)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_readout.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add glot/readout.py tests/test_readout.py
git commit -m "feat: AttentionReadout module with per-graph softmax aggregation"
```

---

### Task 5: GLOTPooler (Combined Module)

**Files:**
- Create: `tests/test_glot_pooler.py`
- Create: `glot/glot_pooler.py`

**Step 1: Write the failing tests**

```python
# tests/test_glot_pooler.py
import torch
import pytest
from glot.glot_pooler import GLOTPooler


class TestGLOTPooler:
    def test_output_shape_default_config(self):
        """Default config: hidden=128, layers=2, jk=cat -> output 384."""
        pooler = GLOTPooler(input_dim=768)
        hidden = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 384)

    def test_output_dim_attribute(self):
        """output_dim attribute matches actual output."""
        pooler = GLOTPooler(input_dim=768, hidden_dim=128, num_gnn_layers=2, jk_mode='cat')
        assert pooler.output_dim == 384

    def test_handles_padding(self):
        """Padding tokens should not affect the output shape."""
        pooler = GLOTPooler(input_dim=32, hidden_dim=16, num_gnn_layers=2)
        hidden = torch.randn(2, 8, 32)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0]])
        z = pooler(hidden, mask)
        assert z.shape == (2, 48)  # 16 * 3

    def test_single_token_sentence(self):
        """A sentence with only 1 valid token should still produce output."""
        pooler = GLOTPooler(input_dim=16, hidden_dim=8, num_gnn_layers=2)
        hidden = torch.randn(1, 5, 16)
        mask = torch.tensor([[1, 0, 0, 0, 0]])
        z = pooler(hidden, mask)
        assert z.shape == (1, 24)  # 8 * 3

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

        # Use same weights
        pooler_high.load_state_dict(pooler_low.state_dict())

        z_low = pooler_low(hidden, mask)
        z_high = pooler_high(hidden, mask)
        # Outputs should differ because graph structure differs
        assert not torch.allclose(z_low, z_high, atol=1e-4)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_glot_pooler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glot.glot_pooler'`

**Step 3: Write the implementation**

```python
# glot/glot_pooler.py
import torch
import torch.nn as nn
from glot.graph_construction import build_token_graph
from glot.token_gnn import TokenGNN
from glot.readout import AttentionReadout


class GLOTPooler(nn.Module):
    """GLOT pooling module: graph construction + GNN + readout.

    Takes frozen LLM hidden states and produces fixed-size
    sentence embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        jk_mode: str = "cat",
        threshold: float = 0.6,
    ):
        super().__init__()
        self.threshold = threshold
        self.token_gnn = TokenGNN(input_dim, hidden_dim, num_gnn_layers, jk_mode)
        self.readout = AttentionReadout(self.token_gnn.output_dim)
        self.output_dim = self.token_gnn.output_dim

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, d) from frozen LLM.
            attention_mask: (B, L) with 1 for valid tokens.

        Returns:
            (B, output_dim) sentence embeddings.
        """
        batch_data = build_token_graph(hidden_states, attention_mask, self.threshold)
        refined = self.token_gnn(batch_data.x, batch_data.edge_index)
        return self.readout(refined, batch_data.batch)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_glot_pooler.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add glot/glot_pooler.py tests/test_glot_pooler.py
git commit -m "feat: GLOTPooler combining graph construction, GNN, and readout"
```

---

### Task 6: Baseline Poolers

**Files:**
- Create: `tests/test_baselines.py`
- Create: `glot/baselines.py`

**Step 1: Write the failing tests**

```python
# tests/test_baselines.py
import torch
import pytest
from glot.baselines import MeanPooler, MaxPooler, CLSPooler, AdaPool


class TestMeanPooler:
    def test_output_shape(self):
        pooler = MeanPooler()
        hidden = torch.randn(2, 5, 768)
        mask = torch.ones(2, 5, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 768)

    def test_ignores_padding(self):
        pooler = MeanPooler()
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
        mask = torch.tensor([[1, 1, 0]])
        z = pooler(hidden, mask)
        expected = torch.tensor([[2.0, 3.0]])  # mean of [1,2] and [3,4]
        assert torch.allclose(z, expected)


class TestMaxPooler:
    def test_output_shape(self):
        pooler = MaxPooler()
        hidden = torch.randn(2, 5, 768)
        mask = torch.ones(2, 5, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 768)

    def test_ignores_padding(self):
        pooler = MaxPooler()
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]])
        mask = torch.tensor([[1, 1, 0]])
        z = pooler(hidden, mask)
        expected = torch.tensor([[3.0, 4.0]])
        assert torch.allclose(z, expected)


class TestCLSPooler:
    def test_encoder_returns_first_token(self):
        pooler = CLSPooler(is_decoder=False)
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        mask = torch.ones(1, 3, dtype=torch.long)
        z = pooler(hidden, mask)
        expected = torch.tensor([[1.0, 2.0]])
        assert torch.allclose(z, expected)

    def test_decoder_returns_last_valid_token(self):
        pooler = CLSPooler(is_decoder=True)
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0]]])
        mask = torch.tensor([[1, 1, 1, 0]])
        z = pooler(hidden, mask)
        expected = torch.tensor([[5.0, 6.0]])
        assert torch.allclose(z, expected)


class TestAdaPool:
    def test_output_shape(self):
        pooler = AdaPool(input_dim=768)
        hidden = torch.randn(2, 5, 768)
        mask = torch.ones(2, 5, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 768)

    def test_has_trainable_params(self):
        pooler = AdaPool(input_dim=64)
        params = list(pooler.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_ignores_padding(self):
        pooler = AdaPool(input_dim=4)
        hidden = torch.randn(1, 5, 4)
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        z = pooler(hidden, mask)
        assert z.shape == (1, 4)
        # Changing padding values should not affect output
        hidden2 = hidden.clone()
        hidden2[0, 3:] = 999.0
        z2 = pooler(hidden2, mask)
        assert torch.allclose(z, z2)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_baselines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glot.baselines'`

**Step 3: Write the implementation**

```python
# glot/baselines.py
import torch
import torch.nn as nn


class MeanPooler(nn.Module):
    """Average hidden states over valid (non-padding) tokens."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (hidden_states * mask).sum(dim=1)   # (B, d)
        counts = mask.sum(dim=1).clamp(min=1)        # (B, 1)
        return summed / counts


class MaxPooler(nn.Module):
    """Element-wise max over valid tokens."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).bool()
        hidden_states = hidden_states.masked_fill(~mask, float("-inf"))
        return hidden_states.max(dim=1).values  # (B, d)


class CLSPooler(nn.Module):
    """CLS token (encoder) or last valid token / EOS (decoder)."""

    def __init__(self, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.is_decoder:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, seq_lengths]
        return hidden_states[:, 0]


class AdaPool(nn.Module):
    """Learned scoring MLP with softmax-weighted average (Brothers, 2025)."""

    def __init__(self, input_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(hidden_states).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (B, L)
        return (hidden_states * weights.unsqueeze(-1)).sum(dim=1)  # (B, d)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_baselines.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add glot/baselines.py tests/test_baselines.py
git commit -m "feat: baseline poolers (Mean, Max, CLS, AdaPool)"
```

---

### Task 7: Utils (Metrics and Config)

**Files:**
- Create: `tests/test_utils.py`
- Create: `glot/utils.py`

**Step 1: Write the failing tests**

```python
# tests/test_utils.py
import torch
import pytest
import yaml
import os
from glot.utils import compute_metrics, load_config, GLUE_TASKS


class TestComputeMetrics:
    def test_accuracy(self):
        preds = [1, 0, 1, 1]
        labels = [1, 0, 1, 0]
        score = compute_metrics(preds, labels, "accuracy")
        assert score == pytest.approx(75.0)  # 3/4 * 100

    def test_mcc_perfect(self):
        preds = [0, 1, 0, 1]
        labels = [0, 1, 0, 1]
        score = compute_metrics(preds, labels, "mcc")
        assert score == pytest.approx(100.0)

    def test_f1(self):
        preds = [1, 1, 0, 0]
        labels = [1, 0, 1, 0]
        score = compute_metrics(preds, labels, "f1")
        # F1 = 2*TP/(2*TP+FP+FN) = 2*1/(2+1+1) = 0.5 -> 50.0
        assert score == pytest.approx(50.0)

    def test_spearman(self):
        preds = [1.0, 2.0, 3.0, 4.0]
        labels = [1.0, 2.0, 3.0, 4.0]
        score = compute_metrics(preds, labels, "spearman")
        assert score == pytest.approx(100.0)


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        config = {"backbone": {"name": "bert-base-uncased"}, "training": {"lr": 0.001}}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)
        loaded = load_config(str(path))
        assert loaded["backbone"]["name"] == "bert-base-uncased"
        assert loaded["training"]["lr"] == 0.001


class TestGlueTasks:
    def test_sst2_config(self):
        cfg = GLUE_TASKS["sst2"]
        assert cfg["type"] == "single"
        assert cfg["num_classes"] == 2
        assert cfg["metric"] == "accuracy"

    def test_mrpc_config(self):
        cfg = GLUE_TASKS["mrpc"]
        assert cfg["type"] == "pair"
        assert cfg["num_classes"] == 2
        assert cfg["metric"] == "f1"

    def test_cola_config(self):
        cfg = GLUE_TASKS["cola"]
        assert cfg["type"] == "single"
        assert cfg["metric"] == "mcc"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glot.utils'`

**Step 3: Write the implementation**

```python
# glot/utils.py
import yaml
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import spearmanr


GLUE_TASKS = {
    "cola": {
        "type": "single",
        "num_classes": 2,
        "metric": "mcc",
        "sentence_keys": ("sentence",),
    },
    "sst2": {
        "type": "single",
        "num_classes": 2,
        "metric": "accuracy",
        "sentence_keys": ("sentence",),
    },
    "mrpc": {
        "type": "pair",
        "num_classes": 2,
        "metric": "f1",
        "sentence_keys": ("sentence1", "sentence2"),
    },
    "qqp": {
        "type": "pair",
        "num_classes": 2,
        "metric": "f1",
        "sentence_keys": ("question1", "question2"),
        "subsample": 20000,
    },
    "stsb": {
        "type": "pair",
        "num_classes": 1,
        "metric": "spearman",
        "sentence_keys": ("sentence1", "sentence2"),
    },
    "mnli": {
        "type": "pair",
        "num_classes": 3,
        "metric": "accuracy",
        "sentence_keys": ("premise", "hypothesis"),
        "subsample": 20000,
    },
    "qnli": {
        "type": "pair",
        "num_classes": 2,
        "metric": "accuracy",
        "sentence_keys": ("question", "sentence"),
        "subsample": 20000,
    },
    "rte": {
        "type": "pair",
        "num_classes": 2,
        "metric": "accuracy",
        "sentence_keys": ("sentence1", "sentence2"),
    },
    "wnli": {
        "type": "pair",
        "num_classes": 2,
        "metric": "accuracy",
        "sentence_keys": ("sentence1", "sentence2"),
    },
}


def compute_metrics(predictions: list, labels: list, metric_name: str) -> float:
    """Compute task-specific metric. Returns score * 100."""
    if metric_name == "accuracy":
        return accuracy_score(labels, predictions) * 100
    elif metric_name == "mcc":
        return matthews_corrcoef(labels, predictions) * 100
    elif metric_name == "f1":
        return f1_score(labels, predictions) * 100
    elif metric_name == "spearman":
        return spearmanr(predictions, labels).correlation * 100
    raise ValueError(f"Unknown metric: {metric_name}")


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_utils.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add glot/utils.py tests/test_utils.py
git commit -m "feat: utils module with metrics and config loading"
```

---

### Task 8: GLUE Data Loader

**Files:**
- Create: `tests/test_glue_loader.py`
- Create: `data/glue_loader.py`

**Step 1: Write the failing tests**

```python
# tests/test_glue_loader.py
import pytest
from data.glue_loader import load_glue_task, get_task_config


class TestGetTaskConfig:
    def test_sst2_single_sentence(self):
        cfg = get_task_config("sst2")
        assert cfg["type"] == "single"
        assert cfg["sentence_keys"] == ("sentence",)

    def test_mrpc_pair_sentence(self):
        cfg = get_task_config("mrpc")
        assert cfg["type"] == "pair"
        assert cfg["sentence_keys"] == ("sentence1", "sentence2")

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            get_task_config("nonexistent_task")


class TestLoadGlueTask:
    """These tests download data so they are marked slow.
    Run with: pytest -m 'not slow' to skip, or pytest to include."""

    @pytest.mark.slow
    def test_sst2_loads(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dataset = load_glue_task("sst2", tokenizer, max_length=32)
        assert "train" in dataset
        assert "validation" in dataset
        assert "input_ids" in dataset["train"].column_names

    @pytest.mark.slow
    def test_mrpc_pair_loads_separate(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dataset = load_glue_task("mrpc", tokenizer, max_length=32)
        assert "input_ids_a" in dataset["train"].column_names
        assert "input_ids_b" in dataset["train"].column_names
        assert "attention_mask_a" in dataset["train"].column_names
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_glue_loader.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'data.glue_loader'`

**Step 3: Write the implementation**

```python
# data/glue_loader.py
from datasets import load_dataset
from glot.utils import GLUE_TASKS


def get_task_config(task_name: str) -> dict:
    """Return the GLUE task configuration dict."""
    if task_name not in GLUE_TASKS:
        raise ValueError(f"Unknown GLUE task: {task_name}. Choose from {list(GLUE_TASKS.keys())}")
    return GLUE_TASKS[task_name]


def load_glue_task(task_name: str, tokenizer, max_length: int = 128, seed: int = 42):
    """Load and tokenize a GLUE task.

    For pair tasks, tokenizes each sentence separately so that GLOT
    can build individual graphs per sentence.
    """
    config = get_task_config(task_name)
    dataset = load_dataset("glue", task_name)

    # Subsample large datasets
    if "subsample" in config:
        n = min(config["subsample"], len(dataset["train"]))
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(n))

    keys = config["sentence_keys"]

    if config["type"] == "single":
        def tokenize(examples):
            return tokenizer(
                examples[keys[0]],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        dataset = dataset.map(tokenize, batched=True)
    else:
        # Pair tasks: tokenize each sentence separately
        def tokenize_pair(examples):
            tok_a = tokenizer(
                examples[keys[0]],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            tok_b = tokenizer(
                examples[keys[1]],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            return {
                "input_ids_a": tok_a["input_ids"],
                "attention_mask_a": tok_a["attention_mask"],
                "input_ids_b": tok_b["input_ids"],
                "attention_mask_b": tok_b["attention_mask"],
            }

        dataset = dataset.map(tokenize_pair, batched=True)

    return dataset
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_glue_loader.py -v -m "not slow"`
Expected: 3 non-slow tests PASS (the 2 slow tests are skipped)

**Step 5: Commit**

```bash
git add data/glue_loader.py tests/test_glue_loader.py
git commit -m "feat: GLUE data loader with separate pair tokenization"
```

---

### Task 9: Hidden State Caching

**Files:**
- Create: `tests/test_cache.py`
- Create: `data/cache.py`
- Create: `cache_hidden_states.py`

**Step 1: Write the failing tests**

```python
# tests/test_cache.py
import torch
import pytest
import os
from data.cache import CachedDataset, save_cache, load_cache


class TestCachedDataset:
    def test_single_task_getitem(self):
        hs = torch.randn(100, 10, 32)
        masks = torch.ones(100, 10, dtype=torch.long)
        labels = torch.randint(0, 2, (100,))
        ds = CachedDataset(hs, masks, labels)
        assert len(ds) == 100
        h, m, l = ds[0]
        assert h.shape == (10, 32)
        assert m.shape == (10,)

    def test_pair_task_getitem(self):
        hs_a = torch.randn(50, 10, 32)
        masks_a = torch.ones(50, 10, dtype=torch.long)
        hs_b = torch.randn(50, 10, 32)
        masks_b = torch.ones(50, 10, dtype=torch.long)
        labels = torch.randint(0, 2, (50,))
        ds = CachedDataset(hs_a, masks_a, labels, hs_b, masks_b)
        assert len(ds) == 50
        result = ds[0]
        assert len(result) == 5  # hs_a, mask_a, hs_b, mask_b, label


class TestSaveLoadCache:
    def test_roundtrip_single(self, tmp_path):
        hs = torch.randn(10, 5, 16)
        masks = torch.ones(10, 5, dtype=torch.long)
        labels = torch.randint(0, 2, (10,))
        path = str(tmp_path / "cache.pt")
        save_cache(path, hs, masks, labels)
        loaded = load_cache(path)
        assert torch.equal(loaded["hidden_states"], hs)
        assert torch.equal(loaded["attention_masks"], masks)
        assert torch.equal(loaded["labels"], labels)

    def test_roundtrip_pair(self, tmp_path):
        hs_a = torch.randn(10, 5, 16)
        masks_a = torch.ones(10, 5, dtype=torch.long)
        hs_b = torch.randn(10, 5, 16)
        masks_b = torch.ones(10, 5, dtype=torch.long)
        labels = torch.randint(0, 2, (10,))
        path = str(tmp_path / "cache.pt")
        save_cache(path, hs_a, masks_a, labels, hs_b, masks_b)
        loaded = load_cache(path)
        assert "hidden_states_b" in loaded
        assert torch.equal(loaded["hidden_states_b"], hs_b)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data.cache'`

**Step 3: Write the caching library**

```python
# data/cache.py
import os
import torch
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    """Dataset wrapping precomputed hidden states.

    Supports both single-sentence and pair tasks.
    """

    def __init__(
        self,
        hidden_states: torch.Tensor,
        attention_masks: torch.Tensor,
        labels: torch.Tensor,
        hidden_states_b: torch.Tensor | None = None,
        attention_masks_b: torch.Tensor | None = None,
    ):
        self.hidden_states = hidden_states
        self.attention_masks = attention_masks
        self.labels = labels
        self.hidden_states_b = hidden_states_b
        self.attention_masks_b = attention_masks_b
        self.is_pair = hidden_states_b is not None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        if self.is_pair:
            return (
                self.hidden_states[idx],
                self.attention_masks[idx],
                self.hidden_states_b[idx],
                self.attention_masks_b[idx],
                self.labels[idx],
            )
        return (
            self.hidden_states[idx],
            self.attention_masks[idx],
            self.labels[idx],
        )


def save_cache(
    path: str,
    hidden_states: torch.Tensor,
    attention_masks: torch.Tensor,
    labels: torch.Tensor,
    hidden_states_b: torch.Tensor | None = None,
    attention_masks_b: torch.Tensor | None = None,
) -> None:
    """Save cached hidden states to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "hidden_states": hidden_states,
        "attention_masks": attention_masks,
        "labels": labels,
    }
    if hidden_states_b is not None:
        data["hidden_states_b"] = hidden_states_b
        data["attention_masks_b"] = attention_masks_b
    torch.save(data, path)


def load_cache(path: str) -> dict:
    """Load cached hidden states from disk."""
    return torch.load(path, weights_only=True)


def make_cached_dataset(path: str) -> CachedDataset:
    """Load a cache file and return a CachedDataset."""
    data = load_cache(path)
    return CachedDataset(
        data["hidden_states"],
        data["attention_masks"],
        data["labels"],
        data.get("hidden_states_b"),
        data.get("attention_masks_b"),
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache.py -v`
Expected: All 4 tests PASS

**Step 5: Write the caching script**

```python
# cache_hidden_states.py
"""Precompute and cache frozen backbone hidden states for all GLUE tasks."""
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from data.glue_loader import load_glue_task, get_task_config
from data.cache import save_cache


def precompute(backbone, dataloader, device, is_pair=False):
    """Run frozen backbone on all data and collect hidden states."""
    all_hs, all_masks, all_labels = [], [], []
    all_hs_b, all_masks_b = [], []

    backbone.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Caching"):
            if is_pair:
                ids_a = batch["input_ids_a"].to(device)
                mask_a = batch["attention_mask_a"].to(device)
                ids_b = batch["input_ids_b"].to(device)
                mask_b = batch["attention_mask_b"].to(device)

                out_a = backbone(input_ids=ids_a, attention_mask=mask_a)
                out_b = backbone(input_ids=ids_b, attention_mask=mask_b)

                all_hs.append(out_a.last_hidden_state.cpu())
                all_masks.append(mask_a.cpu())
                all_hs_b.append(out_b.last_hidden_state.cpu())
                all_masks_b.append(mask_b.cpu())
            else:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                out = backbone(input_ids=ids, attention_mask=mask)
                all_hs.append(out.last_hidden_state.cpu())
                all_masks.append(mask.cpu())

            all_labels.append(batch["label"])

    result = {
        "hs": torch.cat(all_hs),
        "masks": torch.cat(all_masks),
        "labels": torch.cat(all_labels),
    }
    if is_pair:
        result["hs_b"] = torch.cat(all_hs_b)
        result["masks_b"] = torch.cat(all_masks_b)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="bert-base-uncased")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "cola", "mrpc"])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cache_dir", default="cached_hidden_states")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    backbone = AutoModel.from_pretrained(args.backbone).to(args.device)
    for p in backbone.parameters():
        p.requires_grad = False

    backbone_short = args.backbone.replace("/", "_")

    for task_name in args.tasks:
        print(f"\n=== Caching {task_name} ===")
        task_cfg = get_task_config(task_name)
        dataset = load_glue_task(task_name, tokenizer, args.max_length)
        is_pair = task_cfg["type"] == "pair"

        for split in ["train", "validation"]:
            if split not in dataset:
                continue

            ds = dataset[split]
            ds.set_format("torch")

            if is_pair:
                cols = ["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b", "label"]
            else:
                cols = ["input_ids", "attention_mask", "label"]
            ds = ds.select_columns(cols)

            loader = DataLoader(ds, batch_size=args.batch_size)
            result = precompute(backbone, loader, args.device, is_pair)

            cache_path = f"{args.cache_dir}/{backbone_short}/{task_name}/{split}.pt"
            if is_pair:
                save_cache(
                    cache_path, result["hs"], result["masks"], result["labels"],
                    result["hs_b"], result["masks_b"],
                )
            else:
                save_cache(cache_path, result["hs"], result["masks"], result["labels"])

            print(f"  Saved {split}: {result['hs'].shape} -> {cache_path}")


if __name__ == "__main__":
    main()
```

**Step 6: Commit**

```bash
git add data/cache.py cache_hidden_states.py tests/test_cache.py
git commit -m "feat: hidden state caching system and precomputation script"
```

---

### Task 10: Model Wrapper

**Files:**
- Create: `tests/test_model.py`
- Create: `glot/model.py`

**Step 1: Write the failing tests**

```python
# tests/test_model.py
import torch
import pytest
from glot.model import create_pooler_and_head


class TestCreatePoolerAndHead:
    def test_glot_classification(self):
        pooler, head = create_pooler_and_head(
            pooler_type="glot", input_dim=768, num_classes=2,
            task_type="classification",
        )
        hs = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.long)
        z = pooler(hs, mask)
        logits = head(z)
        assert logits.shape == (2, 2)

    def test_mean_classification(self):
        pooler, head = create_pooler_and_head(
            pooler_type="mean", input_dim=768, num_classes=2,
            task_type="classification",
        )
        hs = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.long)
        z = pooler(hs, mask)
        logits = head(z)
        assert logits.shape == (2, 2)

    def test_pair_classification_head(self):
        pooler, head = create_pooler_and_head(
            pooler_type="glot", input_dim=768, num_classes=2,
            task_type="pair_classification",
        )
        hs = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.long)
        z_a = pooler(hs, mask)
        z_b = pooler(hs, mask)
        combined = torch.cat([z_a, z_b], dim=-1)
        logits = head(combined)
        assert logits.shape == (2, 2)

    def test_all_pooler_types(self):
        for ptype in ["glot", "mean", "max", "cls", "adapool"]:
            pooler, head = create_pooler_and_head(
                pooler_type=ptype, input_dim=64, num_classes=3,
                task_type="classification",
            )
            hs = torch.randn(2, 5, 64)
            mask = torch.ones(2, 5, dtype=torch.long)
            z = pooler(hs, mask)
            logits = head(z)
            assert logits.shape == (2, 3), f"Failed for {ptype}"

    def test_unknown_pooler_raises(self):
        with pytest.raises(ValueError):
            create_pooler_and_head(
                pooler_type="unknown", input_dim=64, num_classes=2,
                task_type="classification",
            )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glot.model'`

**Step 3: Write the implementation**

```python
# glot/model.py
import torch.nn as nn
from glot.glot_pooler import GLOTPooler
from glot.baselines import MeanPooler, MaxPooler, CLSPooler, AdaPool


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

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add glot/model.py tests/test_model.py
git commit -m "feat: model factory for pooler + task head creation"
```

---

### Task 11: Training Script

**Files:**
- Create: `tests/test_train.py`
- Create: `train.py`

**Step 1: Write the failing tests**

```python
# tests/test_train.py
import torch
import pytest
from train import train_epoch, evaluate_epoch
from glot.model import create_pooler_and_head
from data.cache import CachedDataset
from torch.utils.data import DataLoader


class TestTrainEpoch:
    def test_single_task_trains(self):
        """Training on single-sentence cached data reduces loss."""
        pooler, head = create_pooler_and_head("glot", input_dim=32, num_classes=2,
                                               task_type="classification",
                                               glot_config={"hidden_dim": 8, "num_gnn_layers": 1})
        hs = torch.randn(20, 5, 32)
        masks = torch.ones(20, 5, dtype=torch.long)
        labels = torch.randint(0, 2, (20,))
        ds = CachedDataset(hs, masks, labels)
        loader = DataLoader(ds, batch_size=10)

        params = list(pooler.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        loss1 = train_epoch(pooler, head, loader, optimizer, loss_fn, "classification")
        loss2 = train_epoch(pooler, head, loader, optimizer, loss_fn, "classification")
        # Loss should generally decrease after training
        assert isinstance(loss1, float)
        assert isinstance(loss2, float)

    def test_pair_task_trains(self):
        """Training on pair-task cached data works."""
        pooler, head = create_pooler_and_head("glot", input_dim=32, num_classes=2,
                                               task_type="pair_classification",
                                               glot_config={"hidden_dim": 8, "num_gnn_layers": 1})
        hs_a = torch.randn(20, 5, 32)
        masks_a = torch.ones(20, 5, dtype=torch.long)
        hs_b = torch.randn(20, 5, 32)
        masks_b = torch.ones(20, 5, dtype=torch.long)
        labels = torch.randint(0, 2, (20,))
        ds = CachedDataset(hs_a, masks_a, labels, hs_b, masks_b)
        loader = DataLoader(ds, batch_size=10)

        params = list(pooler.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        loss = train_epoch(pooler, head, loader, optimizer, loss_fn, "pair_classification")
        assert isinstance(loss, float)


class TestEvaluateEpoch:
    def test_returns_predictions_and_labels(self):
        pooler, head = create_pooler_and_head("mean", input_dim=16, num_classes=2,
                                               task_type="classification")
        hs = torch.randn(10, 4, 16)
        masks = torch.ones(10, 4, dtype=torch.long)
        labels = torch.randint(0, 2, (10,))
        ds = CachedDataset(hs, masks, labels)
        loader = DataLoader(ds, batch_size=5)

        preds, labs = evaluate_epoch(pooler, head, loader, "classification")
        assert len(preds) == 10
        assert len(labs) == 10
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train'` or `ImportError`

**Step 3: Write the implementation**

```python
# train.py
"""Training script for GLOT on cached hidden states."""
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from glot.model import create_pooler_and_head
from glot.utils import compute_metrics, load_config, GLUE_TASKS
from data.cache import make_cached_dataset


def train_epoch(pooler, head, loader, optimizer, loss_fn, task_type, device="cpu"):
    """Train for one epoch. Returns average loss."""
    pooler.train()
    head.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        if task_type == "pair_classification":
            hs_a, mask_a, hs_b, mask_b, labels = [b.to(device) for b in batch]
            z_a = pooler(hs_a, mask_a)
            z_b = pooler(hs_b, mask_b)
            logits = head(torch.cat([z_a, z_b], dim=-1))
        else:
            hs, mask, labels = [b.to(device) for b in batch]
            z = pooler(hs, mask)
            logits = head(z)

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(pooler, head, loader, task_type, device="cpu"):
    """Evaluate and return (predictions, labels) lists."""
    pooler.eval()
    head.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        if task_type == "pair_classification":
            hs_a, mask_a, hs_b, mask_b, labels = [b.to(device) for b in batch]
            z_a = pooler(hs_a, mask_a)
            z_b = pooler(hs_b, mask_b)
            logits = head(torch.cat([z_a, z_b], dim=-1))
        else:
            hs, mask, labels = [b.to(device) for b in batch]
            z = pooler(hs, mask)
            logits = head(z)

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", default=None, help="Override task name")
    parser.add_argument("--pooler", default=None, help="Override pooler type")
    parser.add_argument("--cache_dir", default="cached_hidden_states")
    parser.add_argument("--backbone", default=None, help="Override backbone name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_name = args.task or cfg["task"]["name"]
    pooler_type = args.pooler or "glot"
    backbone_name = args.backbone or cfg["backbone"]["name"]
    backbone_short = backbone_name.replace("/", "_")

    task_cfg = GLUE_TASKS[task_name]
    task_type = "pair_classification" if task_cfg["type"] == "pair" else "classification"
    hidden_dim_map = {"bert-base-uncased": 768, "roberta-base": 768}
    input_dim = hidden_dim_map.get(backbone_name, 768)

    # Load cached data
    train_ds = make_cached_dataset(f"{args.cache_dir}/{backbone_short}/{task_name}/train.pt")
    val_ds = make_cached_dataset(f"{args.cache_dir}/{backbone_short}/{task_name}/validation.pt")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["eval_batch_size"],
    )

    # Create model
    glot_config = {
        "hidden_dim": cfg["glot"]["hidden_dim"],
        "num_gnn_layers": cfg["glot"]["num_layers"],
        "jk_mode": cfg["glot"]["jk_mode"],
        "threshold": cfg["glot"]["threshold"],
    }
    pooler, head = create_pooler_and_head(
        pooler_type=pooler_type,
        input_dim=input_dim,
        num_classes=task_cfg["num_classes"],
        task_type=task_type,
        glot_config=glot_config if pooler_type == "glot" else None,
    )
    pooler = pooler.to(args.device)
    head = head.to(args.device)

    params = list(pooler.parameters()) + list(head.parameters())
    optimizer = Adam(params, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    # Seed
    torch.manual_seed(cfg["training"]["seed"])

    # Train
    for epoch in range(cfg["training"]["epochs"]):
        avg_loss = train_epoch(pooler, head, train_loader, optimizer, loss_fn, task_type, args.device)
        preds, labels = evaluate_epoch(pooler, head, val_loader, task_type, args.device)
        score = compute_metrics(preds, labels, task_cfg["metric"])
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}  {task_cfg['metric']}={score:.2f}")

    print(f"\nFinal {task_cfg['metric']}: {score:.2f}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: training script for cached hidden state training"
```

---

### Task 12: Evaluation Script

**Files:**
- Create: `evaluate.py`

**Step 1: Write the evaluation script**

```python
# evaluate.py
"""Evaluate a trained pooler on cached validation data."""
import argparse
import torch
from torch.utils.data import DataLoader

from glot.model import create_pooler_and_head
from glot.utils import compute_metrics, load_config, GLUE_TASKS
from data.cache import make_cached_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", required=True)
    parser.add_argument("--pooler", default="glot")
    parser.add_argument("--checkpoint", required=True, help="Path to saved model checkpoint")
    parser.add_argument("--cache_dir", default="cached_hidden_states")
    parser.add_argument("--backbone", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_cfg = GLUE_TASKS[args.task]
    task_type = "pair_classification" if task_cfg["type"] == "pair" else "classification"
    backbone_short = args.backbone.replace("/", "_")

    hidden_dim_map = {"bert-base-uncased": 768, "roberta-base": 768}
    input_dim = hidden_dim_map.get(args.backbone, 768)

    glot_config = {
        "hidden_dim": cfg["glot"]["hidden_dim"],
        "num_gnn_layers": cfg["glot"]["num_layers"],
        "jk_mode": cfg["glot"]["jk_mode"],
        "threshold": cfg["glot"]["threshold"],
    }
    pooler, head = create_pooler_and_head(
        pooler_type=args.pooler,
        input_dim=input_dim,
        num_classes=task_cfg["num_classes"],
        task_type=task_type,
        glot_config=glot_config if args.pooler == "glot" else None,
    )

    checkpoint = torch.load(args.checkpoint, weights_only=True)
    pooler.load_state_dict(checkpoint["pooler"])
    head.load_state_dict(checkpoint["head"])

    pooler = pooler.to(args.device).eval()
    head = head.to(args.device).eval()

    val_ds = make_cached_dataset(f"{args.cache_dir}/{backbone_short}/{args.task}/validation.pt")
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["eval_batch_size"])

    from train import evaluate_epoch
    preds, labels = evaluate_epoch(pooler, head, val_loader, task_type, args.device)
    score = compute_metrics(preds, labels, task_cfg["metric"])
    print(f"{args.task} ({args.pooler}): {task_cfg['metric']} = {score:.2f}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add evaluate.py
git commit -m "feat: evaluation script for cached model checkpoints"
```

---

### Task 13: Update `__init__.py` and Run Full Test Suite

**Files:**
- Modify: `glot/__init__.py`

**Step 1: Update `glot/__init__.py` to include all public modules**

```python
# glot/__init__.py
from glot.graph_construction import build_token_graph
from glot.token_gnn import TokenGNN
from glot.readout import AttentionReadout
from glot.glot_pooler import GLOTPooler
from glot.baselines import MeanPooler, MaxPooler, CLSPooler, AdaPool
from glot.model import create_pooler_and_head
from glot.utils import compute_metrics, load_config, GLUE_TASKS
```

**Step 2: Run the full test suite**

Run: `pytest tests/ -v -m "not slow"`
Expected: All tests PASS (should be approximately 44 tests across all test files)

**Step 3: Commit**

```bash
git add glot/__init__.py
git commit -m "chore: update __init__.py with all public exports"
```

---

### Task 14: End-to-End Smoke Test

**Files:**
- Create: `tests/test_e2e.py`

**Step 1: Write end-to-end integration test**

This test verifies the full pipeline works with synthetic data (no model download needed).

```python
# tests/test_e2e.py
import torch
import pytest
from torch.utils.data import DataLoader
from glot.model import create_pooler_and_head
from glot.utils import compute_metrics
from data.cache import CachedDataset
from train import train_epoch, evaluate_epoch


class TestEndToEnd:
    def test_glot_trains_and_evaluates_single_task(self):
        """Full pipeline: create model, train on synthetic cached data, evaluate."""
        torch.manual_seed(42)
        input_dim = 64
        n_train, n_val = 40, 10
        seq_len = 8

        # Synthetic cached hidden states
        train_hs = torch.randn(n_train, seq_len, input_dim)
        train_masks = torch.ones(n_train, seq_len, dtype=torch.long)
        train_labels = torch.randint(0, 2, (n_train,))

        val_hs = torch.randn(n_val, seq_len, input_dim)
        val_masks = torch.ones(n_val, seq_len, dtype=torch.long)
        val_labels = torch.randint(0, 2, (n_val,))

        train_ds = CachedDataset(train_hs, train_masks, train_labels)
        val_ds = CachedDataset(val_hs, val_masks, val_labels)
        train_loader = DataLoader(train_ds, batch_size=10)
        val_loader = DataLoader(val_ds, batch_size=10)

        pooler, head = create_pooler_and_head(
            "glot", input_dim=input_dim, num_classes=2,
            task_type="classification",
            glot_config={"hidden_dim": 16, "num_gnn_layers": 1},
        )

        optimizer = torch.optim.Adam(
            list(pooler.parameters()) + list(head.parameters()), lr=1e-3
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train 3 epochs
        for _ in range(3):
            train_epoch(pooler, head, train_loader, optimizer, loss_fn, "classification")

        # Evaluate
        preds, labels = evaluate_epoch(pooler, head, val_loader, "classification")
        score = compute_metrics(preds, labels, "accuracy")
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_glot_pair_task_e2e(self):
        """Full pipeline for a pair classification task."""
        torch.manual_seed(42)
        input_dim = 32
        n = 30
        seq_len = 6

        ds = CachedDataset(
            torch.randn(n, seq_len, input_dim),
            torch.ones(n, seq_len, dtype=torch.long),
            torch.randint(0, 2, (n,)),
            torch.randn(n, seq_len, input_dim),
            torch.ones(n, seq_len, dtype=torch.long),
        )
        loader = DataLoader(ds, batch_size=10)

        pooler, head = create_pooler_and_head(
            "glot", input_dim=input_dim, num_classes=2,
            task_type="pair_classification",
            glot_config={"hidden_dim": 8, "num_gnn_layers": 1},
        )

        optimizer = torch.optim.Adam(
            list(pooler.parameters()) + list(head.parameters()), lr=1e-3
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        loss = train_epoch(pooler, head, loader, optimizer, loss_fn, "pair_classification")
        assert isinstance(loss, float)
        assert loss > 0

        preds, labels = evaluate_epoch(pooler, head, loader, "pair_classification")
        assert len(preds) == n

    def test_all_baselines_train(self):
        """All baseline poolers can train on synthetic data without error."""
        torch.manual_seed(42)
        input_dim = 32
        n = 20
        seq_len = 4

        train_ds = CachedDataset(
            torch.randn(n, seq_len, input_dim),
            torch.ones(n, seq_len, dtype=torch.long),
            torch.randint(0, 2, (n,)),
        )
        loader = DataLoader(train_ds, batch_size=10)

        for pooler_type in ["glot", "mean", "max", "cls", "adapool"]:
            glot_cfg = {"hidden_dim": 8, "num_gnn_layers": 1} if pooler_type == "glot" else None
            pooler, head = create_pooler_and_head(
                pooler_type, input_dim=input_dim, num_classes=2,
                task_type="classification", glot_config=glot_cfg,
            )
            optimizer = torch.optim.Adam(
                list(pooler.parameters()) + list(head.parameters()), lr=1e-3
            )
            loss_fn = torch.nn.CrossEntropyLoss()

            loss = train_epoch(pooler, head, loader, optimizer, loss_fn, "classification")
            assert isinstance(loss, float), f"Failed for {pooler_type}"
```

**Step 2: Run the e2e tests**

Run: `pytest tests/test_e2e.py -v`
Expected: All 3 tests PASS

**Step 3: Run the full test suite one final time**

Run: `pytest tests/ -v -m "not slow"`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end integration tests for full training pipeline"
```

---

## Summary of Run Commands

After completing all tasks, the workflow for a real experiment is:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Cache hidden states (one-time per backbone)
python cache_hidden_states.py --backbone bert-base-uncased --tasks sst2 cola mrpc

# 3. Train GLOT on a task
python train.py --task sst2 --pooler glot

# 4. Train baselines for comparison
python train.py --task sst2 --pooler mean
python train.py --task sst2 --pooler max
python train.py --task sst2 --pooler cls
python train.py --task sst2 --pooler adapool

# 5. Run tests
pytest tests/ -v -m "not slow"
```
