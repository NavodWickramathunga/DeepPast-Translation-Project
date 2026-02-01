import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


def clean_source(text: str) -> str:
    """
    Cleaning rules aligned with competition formatting tips:
    - normalize whitespace
    - remove certain modern scribal marks: ! ? / : .
    - remove bracket wrappers while keeping content
    - remove half brackets ˹ ˺
    """
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()

    text = text.replace("!", "").replace("?", "")
    text = text.replace("/", " ")
    text = text.replace(":", " ").replace(".", " ")

    # [KÙ.BABBAR] -> KÙ.BABBAR
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)

    # remove half brackets
    text = text.replace("˹", "").replace("˺", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_prefix(text: str) -> str:
    return "translate Akkadian to English: " + text


@dataclass
class GenerationParams:
    max_source_len: int = 160
    max_new_tokens: int = 128
    num_beams: int = 4


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def patch_tokenizer_config_if_needed(model_dir: Path) -> None:
    """
    Fixes a common issue where tokenizer_config.json may contain
    extra_special_tokens as a list (should be dict). This mirrors the Kaggle fix.
    Safe to run even if not needed.
    """
    import json

    cfg_path = model_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        return

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return

    x = cfg.get("extra_special_tokens", None)
    if isinstance(x, list):
        cfg["extra_special_tokens"] = {}
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_model(model_dir: str) -> Tuple[AutoTokenizer, T5ForConditionalGeneration, torch.device]:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model folder not found: {model_path.resolve()}")

    patch_tokenizer_config_if_needed(model_path)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True)

    device = get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


@torch.inference_mode()
def translate_batch(
    tokenizer,
    model,
    device: torch.device,
    texts: List[str],
    params: GenerationParams,
    batch_size: int = 16,
) -> Tuple[List[str], float]:
    """
    Returns (translations, seconds_elapsed)
    """
    t0 = time.time()
    preds: List[str] = []

    prepared = [add_prefix(clean_source(t)) for t in texts]

    for i in range(0, len(prepared), batch_size):
        batch = prepared[i : i + batch_size]
        enc = tokenizer(
            batch,
            max_length=params.max_source_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        out_ids = model.generate(
            **enc,
            max_new_tokens=params.max_new_tokens,
            num_beams=params.num_beams,
            early_stopping=True,
        )
        preds.extend(tokenizer.batch_decode(out_ids, skip_special_tokens=True))

    return preds, (time.time() - t0)
