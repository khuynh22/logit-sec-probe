# logit-sec-probe
Statistical evaluation harness that analyzes LLM token entropy and log-probabilities to detect silent model uncertainty during insecure code generation.

## Quick Start: Google Colab

Run the interactive tutorial directly in your browser - no installation required!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khuynh22/logit-sec-probe/blob/main/entropy_analysis_tutorial.ipynb)

## Installation

### Option 1: Docker (Recommended)

No local installation required! Just use Docker:

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build and run manually
docker build -t logit-sec-probe .
docker run -v ./output:/app/output logit-sec-probe
```

Output files will be saved to the `./output` directory.

#### GPU Support

To enable GPU acceleration, uncomment the GPU section in `docker-compose.yml` and ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

### Option 2: Local Installation

```bash
pip install -r requirements.txt
```

## Usage

### With Docker

```bash
docker-compose up
```

### Local

Run the entropy analysis script:

```bash
python entropy_analysis.py
```

This script:
1. Loads the Qwen/Qwen2.5-Coder-1.5B-Instruct model
2. Generates code for the prompt "Write C function to copy string"
3. Calculates entropy for each generated token
4. Creates a Pandas DataFrame with token analysis
5. Generates a heatmap visualization of token uncertainty

## Output

- `token_entropy_analysis.csv`: CSV file with token-level entropy and probability data
- `token_entropy_heatmap.png`: Heatmap visualization of token uncertainty
