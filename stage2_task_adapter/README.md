Stage-2 task adaptation for C2S-Scale-Gemma-2-2B lives here.

This stage is completion-style supervised fine-tuning for curated cell-sentence-grounded QnA. It is designed to sit on top of the stage-1 domain-fit adapter without overwriting the stage-1 artifact.

Implementation choices:

- completion-only loss, not chat/instruct formatting
- frozen stage-1 adapter kept separate and active during stage-2 forward passes
- stage-2 LoRA trained as a separate adapter on later layers by default
- optional replay mixing from the domain cell-type dataset to reduce forgetting

Recommended launch:

```bash
python train_stage2_task_adapter.py \
  --train-data /path/to/stage2_task_train.jsonl \
  --val-data /path/to/stage2_task_val.jsonl \
  --stage1-adapter /path/to/stage1_domain_adapter/outputs/<run>/checkpoints/best/gemma_lora_stage1 \
  --output-dir /path/to/stage2_outputs \
  --replay-ratio 0.2 \
  --layer-strategy last_third
```

DeepSpeed launch:

```bash
deepspeed --num_gpus 2 train_stage2_task_adapter.py \
  --train-data /path/to/stage2_task_train.jsonl \
  --val-data /path/to/stage2_task_val.jsonl \
  --stage1-adapter /path/to/stage1_domain_adapter/outputs/<run>/checkpoints/best/gemma_lora_stage1 \
  --output-dir /path/to/stage2_outputs \
  --use-deepspeed \
  --zero-stage 2 \
  --layer-strategy last_third
```

Regularized stage-2 launch:

```bash
deepspeed --num_gpus 2 train_stage2_task_adapter_regularized.py \
  --train-data /path/to/stage2_task_train.jsonl \
  --val-data /path/to/stage2_task_val.jsonl \
  --stage1-adapter /path/to/stage1_domain_adapter/outputs/<run>/checkpoints/best/gemma_lora_stage1 \
  --output-dir /path/to/stage2_outputs_reg \
  --use-deepspeed \
  --zero-stage 2 \
  --layer-strategy last_third \
  --embedding-anchor-weight 0.05 \
  --embedding-distribution-weight 0.05 \
  --embedding-relation-weight 0.05
```

This variant adds a loss term that compares the stage-2 prompt embedding against the frozen stage-1 prompt embedding on the same batch. It combines:

- per-sample cosine anchoring
- batch mean and variance matching
- pairwise similarity-structure preservation

Inference with both stages:

```bash
python inference_stage2_task_adapter.py \
  --dataset /path/to/stage2_task_val.jsonl \
  --stage1-adapter /path/to/stage1_domain_adapter/outputs/<run>/checkpoints/best/gemma_lora_stage1 \
  --stage2-adapter /path/to/stage2_outputs/adapter
```

Stage-2 JSONL schema

Required fields:

- `task_type`: one of `cell_type`, `marker_evidence`, `differential_diagnosis`, `state_classification`, `tissue_inference`, `perturbation_inference`, `cluster_caption`, `freeform_qa`
- `answer`: completion text the model should generate

One gene source is required:

- `genes`: ordered list of gene symbols
- or `cell_sentence`: space-separated ordered genes
- or `question`: an existing C2S-style prompt/question field from which genes can be extracted

Optional fields:

- `question_text`: natural-language question for the task
- `task_instruction`: short task directive override
- `candidate_labels`: list of answer candidates for differential diagnosis
- `predicted_label`: useful for evidence questions
- `evidence_genes`: list of genes to mention in the prompt context
- `context`: extra grounded context string
- `sample_id`: arbitrary identifier
- `metadata`: arbitrary JSON object

Raw reasoning dataset support

The trainer can also ingest the data-creation JSONL format directly when rows contain:

- `question`
- `answer`
- `validation_ok`
- `structured_output`
- `textual_ground_truth`

When those fields are present, stage-2 training automatically:

- keeps only samples with `validation_ok = true`
- drops `raw_gpt_output`
- strips URLs and markdown citation brackets from `structured_output` and `textual_ground_truth`
- converts the row into a `freeform_qa` sample whose completion starts with the label and then a cleaned rationale
- drops rows whose rationale is still malformed after cleaning

Example samples:

```json
{"sample_id":"ex1","task_type":"cell_type","genes":["IL7R","LTB","MALAT1"],"answer":"CD4 memory T cell"}
{"sample_id":"ex2","task_type":"marker_evidence","cell_sentence":"NKG7 GNLY PRF1 CTSW","predicted_label":"NK cell","question_text":"Which evidence in the gene expression profile supports the predicted cell type?","answer":"The strong expression of NKG7, GNLY, PRF1, and CTSW supports a cytotoxic lymphocyte identity consistent with NK cells."}
{"sample_id":"ex3","task_type":"differential_diagnosis","genes":["MS4A1","CD79A","HLA-DRA"],"candidate_labels":["B cell","Dendritic cell","Plasma cell"],"answer":"B cell"}
```

Outputs:

- `adapter/`: stage-2 task adapter
- `task_manifest.json`: dataset counts and configuration summary
- `training_summary.json`: trainer metrics and run metadata
- `sanity_generation.json`: one generated validation example after training