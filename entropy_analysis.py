#!/usr/bin/env python3
"""
Token Entropy Analysis Script

Analyzes LLM token entropy and log-probabilities to detect model uncertainty
during code generation.
"""

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


def entropy(logits: torch.Tensor) -> float:
    """
    Calculate entropy from logits.
    
    Applies softmax to convert logits to probabilities, then calculates
    entropy as -sum(p * log(p)).
    
    Args:
        logits: Raw logits tensor from the model
        
    Returns:
        Entropy value as a float
    """
    probs = torch.softmax(logits, dim=-1)
    # Add small epsilon to avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    ent = -torch.sum(probs * log_probs).item()
    return ent


def main():
    """Main function to run token entropy analysis."""
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Prepare prompt
    prompt = "Write C function to copy string"
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = prompt
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate with scores
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract generated tokens (excluding input tokens)
    generated_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
    scores = outputs.scores
    
    # Build data for DataFrame
    data = []
    for i, (token_id, score) in enumerate(zip(generated_tokens, scores)):
        # Get token representation using convert_ids_to_tokens for accurate display
        token_text = tokenizer.convert_ids_to_tokens(token_id.item())
        if token_text is None:
            token_text = f"<{token_id.item()}>"
        
        # Calculate entropy from logits
        token_entropy = entropy(score[0])
        
        # Get probability of selected token
        probs = torch.softmax(score[0], dim=-1)
        token_prob = probs[token_id.item()].item()
        
        data.append({
            "position": i,
            "token": token_text,
            "entropy": token_entropy,
            "probability": token_prob
        })
    
    # Build DataFrame
    df = pd.DataFrame(data)
    print("\nToken Analysis DataFrame:")
    print(df.to_string())
    
    # Save DataFrame to CSV
    df.to_csv("token_entropy_analysis.csv", index=False)
    print("\nDataFrame saved to 'token_entropy_analysis.csv'")
    
    # Plot Heatmap
    # Cap figure width to prevent memory issues with long sequences
    max_width = 20
    fig_width = min(max_width, max(12, len(df) * 0.5))
    plt.figure(figsize=(fig_width, 4))
    
    # Create heatmap data (reshape for single row heatmap)
    heatmap_data = df[["entropy"]].T
    
    # Use position-based labels to avoid issues with special characters or duplicates
    # Tokens are shown in the x-axis labels
    sanitized_labels = [
        f"{i}:{t[:10]}" if len(t) > 10 else f"{i}:{t}"
        for i, t in enumerate(df["token"].tolist())
    ]
    heatmap_data.columns = sanitized_labels
    
    # Plot heatmap with tokens on X-axis and entropy as color
    ax = sns.heatmap(
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
    
    # Save plot
    plt.savefig("token_entropy_heatmap.png", dpi=150, bbox_inches="tight")
    print("Heatmap saved to 'token_entropy_heatmap.png'")
    
    plt.show()
    
    return df


if __name__ == "__main__":
    main()
