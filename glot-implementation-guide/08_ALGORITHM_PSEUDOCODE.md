# 08: Complete Algorithm Pseudocode Reference

## Algorithm 1: GLOT Forward Pass (from paper)

This is the complete pseudocode from Algorithm 1 in Appendix B, with annotations.

```
FUNCTION GLOT(H, M):
    # H ∈ R^{B×L×d_in} : Batch of hidden states from frozen LLM
    # M ∈ {0,1}^{B×L}  : Attention mask
    # τ               : Cosine similarity threshold
    # K               : Number of GNN layers
    # Returns: Z ∈ R^{B×d_out} : Batch of sentence embeddings
    
    # ═══════════════════════════════════════════
    # STEP 1: Token Graph Construction
    # ═══════════════════════════════════════════
    Glist ← []
    FOR i = 1 → B:
        H'_i ← H[i, M[i]==1, :]              # Get valid (non-padding) tokens
        S_i ← COSINE_SIMILARITY(H'_i, H'_i)   # Pairwise similarity matrix
        A_i ← (S_i > τ)                        # Binary adjacency via threshold
        edge_index_i ← ADJACENCY_TO_EDGES(A_i) # Convert to COO format
        Glist.APPEND(nodes=H'_i, edges=edge_index_i)
    END FOR
    
    G_batch ← BATCH_GRAPHS(Glist)   # Combine into single PyG Batch
    U_0, edge_index, batch_idx ← G_batch.x, G_batch.edge_index, G_batch.batch
    
    # ═══════════════════════════════════════════
    # STEP 2: Refinement with TOKEN-GNN
    # ═══════════════════════════════════════════
    U_layers ← [U_0]               # Store outputs from all layers
    FOR k = 1 → K:
        U_{k-1} ← U_layers[k-1]
        U_k ← GNN_LAYER_k(U_{k-1}, edge_index)  # GATConv + ReLU
        U_layers.APPEND(U_k)
    END FOR
    
    # ═══════════════════════════════════════════
    # STEP 3: Feature Fusion (Jumping Knowledge)
    # ═══════════════════════════════════════════
    U_fused ← JUMPING_KNOWLEDGE_CONCAT(U_layers)
    # With K=2 and JK='cat': U_fused ∈ R^{N_total × (3 × hidden_dim)}
    
    # ═══════════════════════════════════════════
    # STEP 4: Readout Layer
    # ═══════════════════════════════════════════
    m ← READOUT_MLP(U_fused)                    # Scalar scores per token
    π ← SOFTMAX_BY_GRAPH(m, batch_idx)          # Normalize within each sentence
    Z_pooled ← π ⊙ U_fused                      # Apply attention weights
    Z ← SCATTER_ADD(Z_pooled, batch_idx)         # Aggregate: weighted sum per graph
    
    RETURN Z
END FUNCTION
```

## Mathematical Formulation Summary

### Step 1: Graph Construction
```
G = (V, E) where |V| = L (number of tokens)
S_ij = cos(x_i, x_j) = (x_i · x_j) / (||x_i|| · ||x_j||)
E = {(i,j) : S_ij > τ}
```

### Step 2: GNN Message Passing (per layer ℓ)
```
a_i^(ℓ) = AGGREGATE({h_j^(ℓ) : j ∈ N(i)})      # Eq. 1
h_i^(ℓ+1) = σ(W^(ℓ) · CONCAT(h_i^(ℓ), a_i^(ℓ)))  # Eq. 2

Where:
- N(i) = neighbors of node i in graph G
- W^(ℓ) ∈ R^{p × 2p}
- σ = ReLU
```

### Step 3: Readout
```
m_i = v^T · tanh(W_m · u_i + b_m)    # Scalar importance score
π = softmax(m)                        # Normalized attention weights  
z = Σ_{i=1}^{L} π_i · u_i            # Eq. 3: Weighted sum → sentence vector
```

## Dimension Flow (with default config)

```
Input:
  H ∈ R^{B × L × d}        (e.g., B=32, L=128, d=4096 for Mistral)
  M ∈ {0,1}^{B × L}

After graph construction + input projection:
  U_0 ∈ R^{N_total × 128}   (N_total = sum of valid tokens across batch)

After GNN layer 1:
  U_1 ∈ R^{N_total × 128}

After GNN layer 2:
  U_2 ∈ R^{N_total × 128}

After Jumping Knowledge (cat):
  U_fused ∈ R^{N_total × 384}  (128 × 3 = 384)

After Readout:
  Z ∈ R^{B × 384}            (one vector per sentence)

After task head:
  logits ∈ R^{B × num_classes}
```

## Key Implementation Details NOT in the Main Algorithm

1. **Input Projection**: Before GNN layers, project from LLM dim `d` to GNN dim `p`:
   `H^(0) = X · W_in` where `W_in ∈ R^{d × p}`

2. **Self-loops**: The paper doesn't explicitly mention removing self-loops. Try both.

3. **GAT Attention Heads**: Default PyG GATConv uses 1 head. Paper doesn't specify multi-head.

4. **Batch Processing**: Use PyG's `Batch.from_data_list()` to handle variable-size graphs.

5. **Gradient Flow**: Only flows through GLOT parameters. Backbone gradients are completely disabled.
