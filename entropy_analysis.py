#!/usr/bin/env python3
"""Token Entropy Analysis Script

Analyzes LLM token entropy and log-probabilities to detect model uncertainty
during code generation.
"""

import os
from pathlib import Path

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Output directory - use /app/output in container, current dir otherwise
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", "/app/output" if os.path.exists("/app/output") else ".")
)


def entropy(logits: torch.Tensor) -> tuple[float, torch.Tensor]:
    """
    Calculate entropy from logits.

    Applies softmax to convert logits to probabilities, then calculates
    entropy as -sum(p * log(p)).

    Args:
        logits (torch.Tensor): Raw logits tensor from the model.

    Returns:
        float: Entropy value.
        torch.Tensor: Probability tensor.
    """
    probs = torch.softmax(logits, dim=-1)
    # Add small epsilon to avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    ent = -torch.sum(probs * log_probs).item()
    return ent, probs


def load_model(model_name: str):
    """Load model and tokenizer from Hugging Face."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer


def generate_with_scores(model, tokenizer, prompt: str):
    """Generate text and return output with scores."""
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
    return generated_tokens, outputs.scores


def build_token_dataframe(tokenizer, generated_tokens, scores) -> pd.DataFrame:
    """Build DataFrame with token entropy analysis."""
    data = []
    for i, (token_id, score) in enumerate(zip(generated_tokens, scores)):
        token_list = tokenizer.convert_ids_to_tokens([token_id.item()])
        token_text = token_list[0] if token_list else f"<{token_id.item()}>"
        if token_text is None:
            token_text = f"<{token_id.item()}>"

        token_entropy, probs = entropy(score[0])
        token_prob = probs[token_id.item()].item()

        data.append({
            "position": i,
            "token": token_text,
            "entropy": token_entropy,
            "probability": token_prob
        })

    return pd.DataFrame(data)


def save_heatmap(df: pd.DataFrame, output_path: Path):
    """Generate and save entropy heatmap visualization."""
    max_width = 20
    fig_width = min(max_width, max(12, len(df) * 0.5))
    plt.figure(figsize=(fig_width, 4))

    heatmap_data = df[["entropy"]].T
    sanitized_labels = [
        f"{i}:{t[:10]}" if len(t) > 10 else f"{i}:{t}"
        for i, t in enumerate(df["token"].tolist())
    ]
    heatmap_data.columns = sanitized_labels

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Entropy"},
        xticklabels=True,
        yticklabels=False
    )

    plt.title("Token Entropy Heatmap - Uncertainty Visualization")
    plt.xlabel("Token")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to '{output_path}'")

    if os.environ.get("DISPLAY") or not os.path.exists("/app/output"):
        plt.show()


def main():
    """Main function to run token entropy analysis."""
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    prompt = "Write C function to copy string"

    model, tokenizer = load_model(model_name)
    generated_tokens, scores = generate_with_scores(model, tokenizer, prompt)

    df = build_token_dataframe(tokenizer, generated_tokens, scores)
    print("\nToken Analysis DataFrame:")
    print(df.to_string())

    csv_path = OUTPUT_DIR / "token_entropy_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDataFrame saved to '{csv_path}'")

    save_heatmap(df, OUTPUT_DIR / "token_entropy_heatmap.png")

    return df


if __name__ == "__main__":
    main()
