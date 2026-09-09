# Literature Shortlist And Metric Choices

## Public models and baselines worth comparing

These are the strongest public options that are relevant to single-cell embeddings, but they are not equally practical from the current JSONL ranked-gene format.

### Directly practical from the current stage-1 setup

- `C2S Base`: the unfine-tuned Gemma cell-to-sentence model is the most direct pre-adapter baseline for your LoRA runs.
- `C2S + LoRA`: your dataset-matched stage-1 adapters are the methods of interest, and both feature-space and projection-space embeddings are worth reporting.
- `TF-IDF + SVD`: a simple lexical baseline on ranked genes; weak scientifically, but very cheap and useful as a floor.
- `Rank PCA`: another cheap floor baseline that preserves only ranked-gene structure.
- `SentenceTransformer`: not single-cell-specific, but it gives a public generic language-model baseline over gene sentences.

### Public single-cell foundation models with strong literature support

- `scGPT`: public code and pretrained checkpoints, including a recommended whole-human model and zero-shot cell embedding workflows.
- `Geneformer`: public pretrained models on tens of millions of transcriptomes, strong zero-shot and fine-tuned downstream results, but requires its own package and token dictionaries.
- `UCE`: public zero-shot foundation model focused on AnnData count matrices and external model weights.
- `scVI` or `scANVI`: public probabilistic baselines that are still strong for embedding, integration, and annotation, but they need count-matrix or AnnData inputs rather than ranked-gene JSONL.

## Why this benchmark v1 does not force every literature model into the default run

Your current stage-1 assets are ranked-gene JSONL files plus dataset-specific LoRA checkpoints. That format supports the C2S family and rank-based baselines immediately, while UCE and scVI-style methods want count matrices or h5ad inputs, and scGPT or Geneformer require extra model assets.

For that reason the benchmark suite is split into:

- `implemented and local by default`: rank baselines plus C2S base and C2S LoRA
- `implemented but optional`: scGPT, SentenceTransformer, Geneformer
- `planned, not wired in v1`: UCE and scVI-style baselines that need a separate count-matrix pipeline

## Common evaluation metrics in the literature

Cell embedding papers usually report a mixture of these metric families:

- reference mapping or annotation accuracy
- kNN label transfer accuracy or macro-F1
- clustering metrics such as NMI and ARI
- silhouette or ASW for cell-type separation
- retrieval or nearest-neighbor recall for same-label cells
- batch mixing or integration metrics such as batch ASW, kBET, graph connectivity, cLISI, and iLISI

The benchmark suite implements the metrics that are defensible from the current stage-1 artifacts without rebuilding the datasets into count matrices:

- within-split Recall@K
- train-reference label transfer Recall@K
- weighted kNN transfer accuracy and macro-F1
- NMI
- ARI
- silhouette score

## Why grouped and standard should be reported separately

The standard split is a cell-level benchmark and is easier because similar cells from the same donor-like source can be present across splits. The grouped split is closer to a donor or specimen holdout and is therefore the cleaner generalization benchmark.

For stage-1 adapter evaluation, the most useful reporting pattern is:

- `standard`: sanity check and easier retrieval/transfer setting
- `grouped`: stricter generalization setting

Do not collapse them into one score unless you explicitly want a blended summary.
