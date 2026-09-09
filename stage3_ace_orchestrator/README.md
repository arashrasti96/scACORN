Stage-3 ACE orchestrator runtime for routing queries across stage-2 task adapters.

This package treats the trained stage-2 adapters as local expert tools that can be
discovered from the stage-2 `outputs/` directory, described to an ACE Generator,
and invoked lazily at inference time.

Initial scope in this scaffold:

- manifest-driven expert discovery
- explicit per-expert routing profiles in `expert_profiles.json`
- repo/path normalization for cluster-authored manifests
- runtime configuration shared by the orchestrator and benchmark runner

Planned follow-up modules:

- request normalization from ad hoc queries and stage-3 benchmark rows
- lazy expert inference engine on top of the stage-2 adapter loading pattern
- ACE tool bridge exposing one tool per expert
- benchmark/evaluation runner over `dataset_pipeline/data/stage3_exports`

Implemented entrypoints:

- `run_stage3_query.py`: one-off orchestrated query execution
- `run_stage3_benchmark.py`: benchmark/evaluation over stage-3 JSONL exports
- `train_stage3_playbook.py`: ACE offline playbook training where the Generator
	can call local expert tools, receive their outputs, and update the playbook
	from environment feedback plus ground truth

Example dry-run:

```bash
PYTHONPATH=/path/to/agentic-context-engine:/path/to/agentic-context-engine/benchmarks/src/Train/Contrastive_learning \
python benchmarks/src/Train/Contrastive_learning/stage3_ace_orchestrator/train_stage3_playbook.py \
	--samples benchmarks/src/Train/Contrastive_learning/dataset_pipeline/data/stage3_exports/tissue_inference_direct/grouped_train.jsonl \
	--max-samples 3 \
	--dry-run
```