#!/usr/bin/env python3
"""FastAPI Backend for Token Entropy Analysis

Production-ready API for analyzing LLM token entropy and detecting risky code patterns.
"""
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables for model and tokenizer
model = None
tokenizer = None

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def entropy(logits: torch.Tensor) -> tuple[float, torch.Tensor]:
    """
    Calculate entropy from logits.

    Applies softmax to convert logits to probabilities, then calculates
    entropy as -sum(p * log(p)).

    Args:
        logits (torch.Tensor): Raw logits tensor from the model.

    Returns:
        tuple[float, torch.Tensor]: Entropy value and probability tensor.
    """
    probs = torch.softmax(logits, dim=-1)
    # Add small epsilon to avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    ent = -torch.sum(probs * log_probs).item()
    return ent, probs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for loading model on startup and cleanup on shutdown.
    """
    # Startup: Load model and tokenizer
    global model, tokenizer

    print(f"Loading model: {MODEL_NAME}")
    print("This may take a few minutes on first run...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"✓ Model loaded successfully on {model.device}")
    print(f"  Vocabulary size: {tokenizer.vocab_size:,} tokens")

    yield

    # Shutdown: Cleanup
    print("Shutting down and cleaning up resources...")
    model = None
    tokenizer = None


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Token Entropy Analysis API",
    description="Analyze LLM token-level entropy and detect risky code patterns",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class AnalyzeRequest(BaseModel):
    """Request schema for /analyze endpoint."""
    prompt: str = Field(..., description="User prompt for code generation")
    system_prompt: Optional[str] = Field(
        "You are a secure coding assistant.",
        description="System instruction for the model"
    )
    max_tokens: int = Field(200, ge=1, le=1000, description="Maximum tokens to generate")
    risky_keywords: List[str] = Field(
        ["strcpy", "system", "eval", "exec"],
        description="Keywords to flag as risky in generated tokens"
    )


class TokenDetail(BaseModel):
    """Schema for individual token details."""
    token: str
    entropy: float
    prob: float
    is_risky: bool


class AnalyzeResponse(BaseModel):
    """Response schema for /analyze endpoint."""
    generated_text: str
    tokens: List[TokenDetail]
    metadata: Dict[str, Any]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Token Entropy Analysis API is running",
        "model": MODEL_NAME,
        "device": str(model.device) if model else "not loaded"
    }


@app.get("/health")
async def health():
    """Detailed health check endpoint."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(model.device),
        "vocab_size": tokenizer.vocab_size
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze token-level entropy for LLM code generation.

    This endpoint generates code using the loaded model and returns:
    - Full generated text
    - Per-token entropy and probability
    - Risky keyword detection

    Args:
        request: AnalyzeRequest with prompt, system_prompt, max_tokens, risky_keywords

    Returns:
        AnalyzeResponse with generated text and token-level analysis
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build messages list for chat template
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        # Apply chat template
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for tokenizers without chat template
            if request.system_prompt:
                text = f"System: {request.system_prompt}\n\nUser: {request.prompt}\n\nAssistant:"
            else:
                text = request.prompt

        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate with scores
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Extract generated tokens (excluding input prompt)
        generated_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        scores = outputs.scores

        # Decode full generated text
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Analyze each token
        token_details = []
        risky_keywords_lower = [kw.lower() for kw in request.risky_keywords]

        for i, (token_id, score) in enumerate(zip(generated_tokens, scores)):
            # Get token text
            token_list = tokenizer.convert_ids_to_tokens([token_id.item()])
            token_text = token_list[0] if token_list else f"<{token_id.item()}>"
            if token_text is None:
                token_text = f"<{token_id.item()}>"

            # Calculate entropy and probability
            token_entropy, probs = entropy(score[0])
            token_prob = probs[token_id.item()].item()

            # Check if token contains risky keyword
            token_lower = token_text.lower()
            is_risky = any(keyword in token_lower for keyword in risky_keywords_lower)

            token_details.append(TokenDetail(
                token=token_text,
                entropy=round(token_entropy, 6),
                prob=round(token_prob, 6),
                is_risky=is_risky
            ))

        # Calculate metadata
        risky_count = sum(1 for t in token_details if t.is_risky)
        avg_entropy = sum(t.entropy for t in token_details) / len(token_details) if token_details else 0.0
        avg_prob = sum(t.prob for t in token_details) / len(token_details) if token_details else 0.0

        metadata = {
            "total_tokens": len(token_details),
            "risky_tokens_count": risky_count,
            "avg_entropy": round(avg_entropy, 6),
            "avg_probability": round(avg_prob, 6),
            "max_entropy": round(max((t.entropy for t in token_details), default=0.0), 6),
            "min_probability": round(min((t.prob for t in token_details), default=1.0), 6)
        }

        return AnalyzeResponse(
            generated_text=generated_text,
            tokens=token_details,
            metadata=metadata
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
