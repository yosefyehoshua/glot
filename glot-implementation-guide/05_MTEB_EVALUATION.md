# 05: MTEB Benchmark Evaluation

## Overview
Evaluate GLOT as a general-purpose sentence encoder on 7 diverse tasks from MTEB. Uses a two-stage protocol: (1) train on MS MARCO with contrastive loss, (2) zero-shot evaluation on MTEB tasks.

---

## Stage 1: Training on MS MARCO

### Dataset
- **MS MARCO** passage ranking dataset (Bajaj et al., 2016)
- Task: Given a query, retrieve the relevant passage
- Available via HuggingFace: `ms_marco` or custom loading

### Contrastive Training Setup
```python
from datasets import load_dataset

# Load MS MARCO
# Each example has: query, positive_passage, (optionally hard negatives)
dataset = load_dataset("ms_marco", "v2.1")

# Training config
MTEB_TRAIN_CONFIG = {
    'loss': 'symmetric_contrastive',
    'temperature': 0.07,
    'epochs': 2,  # Same as other experiments
    'learning_rate': 2e-4,
    'batch_size': 32,
    'backbone_frozen': True,
}
```

### Contrastive Loss Implementation
```python
import torch
import torch.nn.functional as F

def symmetric_contrastive_loss(z_queries, z_passages, temperature=0.07):
    """
    Symmetric in-batch contrastive loss.
    
    Each query's positive passage is the diagonal entry.
    All other passages in the batch serve as negatives.
    
    Args:
        z_queries: (B, D) normalized query embeddings
        z_passages: (B, D) normalized passage embeddings
        temperature: scaling temperature (0.07 in paper)
    """
    z_q = F.normalize(z_queries, dim=-1)
    z_p = F.normalize(z_passages, dim=-1)
    
    # Similarity matrix
    sim = z_q @ z_p.T / temperature  # (B, B)
    
    # Labels: diagonal entries are positives
    labels = torch.arange(sim.size(0), device=sim.device)
    
    # Symmetric loss
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss
```

### Training Loop for MTEB
```python
def train_mteb(model, train_loader, config):
    """Train pooler with contrastive loss on MS MARCO."""
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate']
    )
    
    for epoch in range(config['epochs']):
        model.train()
        model.backbone.eval()  # Keep backbone frozen
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Encode queries and passages separately
            z_q = model.encode(batch['query_ids'], batch['query_mask'])
            z_p = model.encode(batch['passage_ids'], batch['passage_mask'])
            
            loss = symmetric_contrastive_loss(z_q, z_p, config['temperature'])
            loss.backward()
            optimizer.step()
```

---

## Stage 2: Zero-Shot MTEB Evaluation

### MTEB Tasks Used in Paper (Table 3)

| Task | Category | Metric | Description |
|------|----------|--------|-------------|
| EmotionClassification | Classification | Accuracy | Multi-class tweet emotion |
| SciFact | Retrieval/Reranking | NDCG@10 | Verify scientific claims |
| RedditClustering | Clustering | V-Measure | Cluster Reddit comments |
| AskUbuntuDupQuestions | Retrieval | MAP | Find duplicate questions |
| STS12 | STS | Cosine Spearman | Semantic similarity |
| TwitterSemEval2015 | PairClassification | Max AP | Paraphrase detection |
| SummEval | Summarization | Cosine Spearman | Summary evaluation |

### Using the MTEB Library
```python
import mteb

# Create a custom model wrapper for MTEB
class GLOTSentenceEncoder:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def encode(self, sentences, batch_size=64, **kwargs):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            encoded = self.tokenizer(
                batch_sentences, truncation=True, max_length=512,
                padding=True, return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    encoded['input_ids'], encoded['attention_mask']
                )
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)

# Run MTEB evaluation
encoder = GLOTSentenceEncoder(trained_model, tokenizer)

tasks = mteb.get_tasks(
    tasks=["EmotionClassification", "SciFact", "RedditClustering",
           "AskUbuntuDupQuestions", "STS12", "TwitterSemEval2015", "SummEval"]
)

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="results/mteb")
```

### Full MTEB Evaluation (Table 12 in Appendix)
The paper also evaluates on the full MTEB benchmark with many more tasks. The comprehensive table includes:
- 12 Classification tasks
- 15 Retrieval tasks (some still running in the paper)
- 11 Clustering tasks
- 10 STS tasks
- 4 Reranking tasks
- 3 PairClassification tasks
- 1 Summarization task

### Metrics Explanation
- **Accuracy**: Standard classification accuracy
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10 (retrieval quality)
- **V-Measure**: Clustering quality (harmonic mean of homogeneity and completeness)
- **MAP**: Mean Average Precision (retrieval)
- **Cosine Spearman**: Spearman correlation of cosine similarity scores
- **Max AP**: Maximum Average Precision (pair classification)

---

## Expected Results (Table 3, selected)

With LLaMA-3B backbone:

| Task | EOS | Mean | Max | AdaPool | GLOT |
|------|-----|------|-----|---------|------|
| EmotionClass. | 0.2765 | 0.2920 | 0.2478 | 0.2185 | **0.3046** |
| SciFact | 0.0087 | 0.4247 | 0.4087 | 0.4140 | **0.4586** |
| AskUbuntu | 0.4420 | 0.4971 | 0.4906 | 0.4946 | **0.5103** |

GLOT achieves the best score on all 7 tasks for RoBERTa, and strong performance across all backbones.
