# OmniVoice Agent Guide

## Repository focus

- `omnivoice/models/omnivoice.py` is the core model implementation.
- User-facing inference entrypoints live in `omnivoice/cli/`.
- Batch manifest parsing lives in `omnivoice/utils/data_utils.py`.
- Documentation should stay in `README.md` or `docs/` unless a new file is explicitly required.

## Memory integration rules

- Memoria retrieval must condition `instruct` or adjacent metadata only.
- Do not rewrite the spoken `text` input when adding memory context.
- Prefer the embedded Memoria layer in `omnivoice/integrations/memoria.py` for local/offline workflows.
- Async memory stores return an `mref_*` receipt immediately and complete in the detached worker module.

## Validation expectations

- Run the existing CLI smoke checks after inference-facing changes:
  - `omnivoice-infer --help`
  - `omnivoice-infer-batch --help`
  - `omnivoice-demo --help`
- Use `python -m compileall omnivoice` for a fast syntax pass across the package.

## Packaging notes

- Core install remains `pip install -e .`
- Memory support is optional via `pip install -e ".[memory]"`
- The memory extra is reserved for ONNX Runtime-backed local embeddings
