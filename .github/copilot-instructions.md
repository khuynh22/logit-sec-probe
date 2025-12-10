# Copilot Instructions for logit-sec-probe

## Project Overview

**logit-sec-probe** is a security research tool that evaluates LLM vulnerability to insecure code generation by analyzing token-level entropy and log-probabilities. The core hypothesis: safety prompts may increase model uncertainty (entropy) when generating risky code patterns.

### Architecture Pattern: A/B Testing Framework

The codebase implements a controlled experiment comparing two configurations:
- **Base**: No system prompt (baseline behavior)
- **Safety**: Safety system prompt enabled ("You are a secure coding assistant...")

Each CWE test case runs through both configs, generating code while capturing internal model uncertainty metrics (entropy, probabilities) for every token.

## Key Components

### 1. Core Analysis Script (`entropy_analysis.py`)

**Critical workflow**: `main()` → `load_cwe_prompts()` → `run_experiment()` → `generate_with_scores()` → `analyze_tokens()` → `plot_comparative_analysis()`

Key functions:
- `entropy(logits)`: Calculates Shannon entropy from model logits using `-sum(p * log(p))` formula
- `generate_with_scores()`: Uses `model.generate()` with `output_scores=True` to capture per-token logits
- `analyze_tokens()`: Tags risky tokens by matching CWE-specific keywords (e.g., `strcpy` for buffer overflow)

**Environment-aware paths**: Uses `/app/output` and `/app/data` when running in Docker, falls back to local `./output` and `./data` otherwise (see `OUTPUT_DIR` and `DATA_DIR` path logic).

### 2. Test Data (`data/cwe_prompts.json`)

Structure: Array of CWE test cases with:
```json
{
  "id": "CWE-XXX",
  "name": "Vulnerability Name", 
  "prompt": "Instruction to generate vulnerable code",
  "risky_keyword": "keyword to flag in generated tokens"
}
```

Current coverage: CWE-120 (Buffer Overflow), CWE-89 (SQL Injection), CWE-79 (XSS).

### 3. Output Artifacts

- `experiment_results.csv`: Token-level data with columns `Experiment_ID`, `Config`, `Token_Pos`, `Token_Text`, `Entropy`, `Probability`, `Is_Risky`
- `comparative_entropy.png`: FacetGrid heatmap (rows=CWE, cols=Base/Safety, color=entropy)

## Developer Workflows

### Running Experiments

**Docker (recommended)**:
```bash
docker-compose up --build
# Results saved to ./output/
```

**Local Python**:
```bash
pip install -r requirements.txt
python entropy_analysis.py
```

**GPU Support**: Uncomment `deploy.resources` section in `docker-compose.yml` (requires NVIDIA Container Toolkit).

### Interactive Development

Use `entropy_analysis_tutorial.ipynb` for step-by-step exploration:
- Load on Google Colab via badge in README (no setup required)
- Demonstrates entropy calculation, model loading, and visualization workflow
- Includes LaTeX math notation for entropy formula

### Adding New CWE Test Cases

1. Add entry to `data/cwe_prompts.json` with unique CWE ID
2. Define `risky_keyword` for token tagging (case-insensitive match)
3. Re-run `python entropy_analysis.py` - visualization automatically expands to new rows

### Modifying Experiment Configs

Edit `CONFIGS` dict in `entropy_analysis.py`:
```python
CONFIGS = {
    "ConfigName": {
        "system_prompt": "Your prompt or None",
        "description": "Human-readable description"
    }
}
```

Visualization columns adapt automatically to number of configs.

## Model Integration

**Default model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (1.5B params, chosen for speed + code capability)

**Chat template handling**: Code checks `tokenizer.apply_chat_template()` availability; falls back to manual formatting if not supported (see `generate_with_scores()` implementation).

**HuggingFace cache**: Docker mounts persistent volume `hf-cache` to avoid re-downloading models (~3GB).

## Code Conventions

- **Type hints**: Use modern Python 3.10+ syntax (`tuple[float, torch.Tensor]` not `Tuple`)
- **Pathlib over os.path**: All file paths use `Path` objects with `/` for cross-platform compatibility
- **Docstring style**: Google-style with Args/Returns sections
- **Tensor handling**: Always check device placement; use `.item()` to extract scalars from tensors
- **Visualization**: Use `plt.close()` after saving to avoid memory leaks in headless environments

## Testing & Validation

**No formal test suite** - validation happens via:
1. Manual inspection of generated code in console output
2. CSV sanity checks (entropy range, risky token counts)
3. Visual verification of heatmap alignment (token labels, color normalization)

**Common debugging**: If `Is_Risky` always False, verify `risky_keyword` matches actual tokenizer output (may be subword tokens like `▁strcpy`).

## Dependencies & Constraints

- Requires **PyTorch ≥2.0** for `torch.float16` and `device_map="auto"`
- **transformers ≥4.35** for `output_scores` and `return_dict_in_generate`
- Minimum 4GB RAM for 1.5B model inference (16GB recommended for larger models)

## Research Context

This tool supports the hypothesis that safety-aligned models exhibit measurable internal uncertainty (higher entropy) when prompted to generate insecure code, even if they comply with the request. Results inform AI safety research on detecting silent model hesitation.
