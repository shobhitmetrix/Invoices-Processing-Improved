"""Standalone batch invoice extraction — Gemini via OpenRouter (Test Run folder).

Single file: invoice schemas/prompts/math-check logic + OpenRouter helpers + batch runner.
No imports from ../extract_invoice.py or ../Model Comparison/compare_models.py.

Prerequisites:
  1. Invoice images in Test Set/ (or pass --images).
  2. OPENROUTER_API_KEY in .env next to this script or in the parent P&P LLM Testing folder.

Usage (from this folder):
    python run_test_set_gemini.py
    python run_test_set_gemini.py --images "Test Set" --resume
    python run_test_set_gemini.py --no-math-check
    python run_test_set_gemini.py --limit 5
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
load_dotenv(HERE.parent / ".env")

DEFAULT_IMAGES = HERE / "Test Set"
DEFAULT_OUTPUT = HERE / "output" / "test_run_gemini_results.json"
DEFAULT_MODELS_CONFIG = HERE / "models_gemini.json"
GEMINI_MODEL_ID = "google/gemini-3.8-flash"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

POST_TAX_TOLERANCE = 0.30
MAX_TRIES = 3
MATH_CHECK_ENABLED = True

RETRY_PROMPT = (
    "This does not add up. Default Price is per unit on the invoice; "
    "only use Amount/Quantity when the rate is per Ltr or per Kg. "
    "If Basic Amt and No of Bag/Pouch columns exist, use that layout for Price/Quantity. "
    "Change Price or Quantity if you picked the wrong column. "
    "Also write one general prompt line that would have avoided this mistake "
    "on the first try. Do not mention this invoice, SKU names, or specific amounts."
)
PROMPT = (
    "Extract data from this paints purchase invoice. "
    "For each line item: Description, Price, Quantity, Amount. "
    "Amount: line total (right-most total); do not change it. Prefer post-tax. "
    "Quantity: number of packs/pieces (bags, pouches, cartons)—not litres, kg, or MT. "
    "Price: per-unit rate on the invoice (Rate/Price/MRP per pack or piece). "
    "Do not replace printed unit Price with Amount÷Quantity unless the row has no pack rate "
    "and the only rate label is per Ltr, Kg, or MT—in that case derive unit Price from Amount÷pack Quantity. "
    "Exception—only if the table has No of Bag/Pouch and Basic Amt columns: "
    "Quantity = No of Bag/Pouch; Price = Basic Amt÷No of Bag/Pouch; ignore Basic/MT/L/KG and Qty in MT/L/KG. "
    "Price×Quantity should match Amount (within 30% if Amount is post-tax). "
    "Say pre_tax or post_tax for each Amount. "
    "Product/SKU rows only; skip CGST/SGST/IGST rows with no SKU. "
    "Use null if missing. List issues when Price×Quantity does not match Amount."
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def set_math_check(enabled: bool) -> None:
    global MATH_CHECK_ENABLED
    MATH_CHECK_ENABLED = enabled


def normalize_prompt_line(line: str | None) -> str | None:
    if not line:
        return None
    text = " ".join(line.split()).strip(" -")
    if len(text) < 20 or len(text) > 240:
        return None
    return text


def to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def to_json(model: BaseModel, indent: int = 2) -> str:
    data = to_dict(model)
    data.pop("issues", None)
    return json.dumps(data, indent=indent)


class LineItem(BaseModel):
    description: str | None = Field(
        default=None,
        description="Product or SKU name. Not a tax label like Central GST or State GST.",
    )
    price: float | None = Field(
        default=None,
        description="Per pack/piece rate from invoice; Basic Amt÷bag count only when those columns exist.",
    )
    quantity: float | None = Field(
        default=None,
        description="Pack/pouch/carton count—not volume or weight (MT/L/KG qty column).",
    )
    amount: float | None = None
    amount_tax_type: Literal["pre_tax", "post_tax"] | None = Field(
        default=None,
        description="Whether this line Amount is before tax or after tax. Prefer post_tax.",
    )


class PaintsInvoice(BaseModel):
    invoice_number: str | None = None
    invoice_date: str | None = None
    supplier_name: str | None = None
    buyer_name: str | None = None
    line_items: list[LineItem]
    total_amount: float | None = None
    total_amount_tax_type: Literal["pre_tax", "post_tax"] | None = Field(
        default=None,
        description="Whether total_amount is before tax or after tax. Prefer post_tax.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Only list issues if price * quantity does not match Amount. Empty otherwise.",
    )


class RetryResult(BaseModel):
    invoice: PaintsInvoice
    prompt_line: str | None = Field(
        default=None,
        description=(
            "One short general rule for the first-pass prompt so this "
            "Price/Quantity column mistake is avoided next time. "
            "No invoice numbers, SKU names, or specific amounts."
        ),
    )


def encode_image(path: Path) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return mime, data


def rule_issues(invoice: PaintsInvoice) -> list[str]:
    issues: list[str] = []
    if not invoice.line_items:
        issues.append("No line items extracted.")
    for i, item in enumerate(invoice.line_items, start=1):
        if item.amount is not None and item.amount_tax_type is None:
            issues.append(f"Line {i}: amount_tax_type is missing.")
        if not MATH_CHECK_ENABLED:
            continue
        if (
            item.price is not None
            and item.quantity is not None
            and item.amount is not None
        ):
            expected = item.price * item.quantity
            if item.amount_tax_type == "pre_tax":
                if abs(item.amount - expected) > 0.05:
                    issues.append(
                        f"Line {i}: pre-tax amount {item.amount} does not match "
                        f"price {item.price} x quantity {item.quantity}."
                    )
            elif item.amount_tax_type == "post_tax":
                if item.amount == 0:
                    issues.append(f"Line {i}: post-tax amount is 0.")
                else:
                    gap = abs(item.amount - expected) / abs(item.amount)
                    if gap > POST_TAX_TOLERANCE:
                        issues.append(
                            f"Line {i}: price x quantity {expected} is not within "
                            f"{int(POST_TAX_TOLERANCE * 100)}% of post-tax amount {item.amount} "
                            f"(gap {gap:.0%})."
                        )
    return issues


def line_math_gap(item: LineItem) -> float | None:
    if item.price is None or item.quantity is None or item.amount is None:
        return None
    if item.amount == 0:
        return 1.0
    expected = item.price * item.quantity
    return abs(item.amount - expected) / abs(item.amount)


def math_is_far(invoice: PaintsInvoice) -> bool:
    for item in invoice.line_items:
        gap = line_math_gap(item)
        if gap is not None and gap > POST_TAX_TOLERANCE:
            return True
    return False


def mismatch_notes(invoice: PaintsInvoice) -> str:
    lines: list[str] = []
    for i, item in enumerate(invoice.line_items, start=1):
        gap = line_math_gap(item)
        if gap is None or gap <= POST_TAX_TOLERANCE:
            continue
        expected = item.price * item.quantity
        lines.append(
            f"Line {i}: price {item.price} x quantity {item.quantity} = {expected}, "
            f"Amount {item.amount}."
        )
    return "\n".join(lines)


def far_line_count(invoice: PaintsInvoice) -> int:
    count = 0
    for item in invoice.line_items:
        gap = line_math_gap(item)
        if gap is not None and gap > POST_TAX_TOLERANCE:
            count += 1
    return count


def collect_images(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(
        p for p in target.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


@dataclass
class ModelSpec:
    label: str
    provider: str
    model: str
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None


class RunUsage:
    def __init__(self, model: str, input_rate: float, output_rate: float) -> None:
        self.model = model
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reported_cost_usd: float | None = None

    def add_openrouter_usage(self, usage: object | None, cost: float | None = None) -> None:
        self.calls += 1
        if usage is None:
            return
        if isinstance(usage, dict):
            self.input_tokens += usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            self.output_tokens += usage.get("completion_tokens") or usage.get("output_tokens") or 0
        else:
            self.input_tokens += getattr(usage, "prompt_tokens", 0) or getattr(
                usage, "input_tokens", 0
            )
            self.output_tokens += getattr(usage, "completion_tokens", 0) or getattr(
                usage, "output_tokens", 0
            )
        if cost is not None:
            self.reported_cost_usd = (self.reported_cost_usd or 0.0) + cost

    def estimated_cost_usd(self) -> float:
        if self.reported_cost_usd is not None:
            return self.reported_cost_usd
        return (
            self.input_tokens / 1_000_000 * self.input_rate
            + self.output_tokens / 1_000_000 * self.output_rate
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "api_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 6),
            "cost_from_provider": self.reported_cost_usd is not None,
        }


def load_model_specs(path: Path) -> list[ModelSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    specs: list[ModelSpec] = []
    for entry in data.get("models", []):
        specs.append(
            ModelSpec(
                label=str(entry.get("label") or entry["model"]),
                provider=str(entry["provider"]).lower(),
                model=str(entry["model"]),
                input_usd_per_million=entry.get("input_usd_per_million"),
                output_usd_per_million=entry.get("output_usd_per_million"),
            )
        )
    if not specs:
        raise ValueError(f"No models listed in {path}")
    return specs


def openrouter_extra_cost(response: object) -> float | None:
    for key in ("cost", "total_cost", "generation_cost"):
        val = getattr(response, key, None)
        if val is not None:
            return float(val)
    extra = getattr(response, "model_extra", None) or {}
    if isinstance(extra, dict):
        for key in ("cost", "total_cost"):
            if key in extra:
                return float(extra[key])
    return None


def rates_for_spec(spec: ModelSpec) -> tuple[float, float]:
    inp = spec.input_usd_per_million if spec.input_usd_per_million is not None else 0.0
    out = spec.output_usd_per_million if spec.output_usd_per_million is not None else 0.0
    if inp == 0 and out == 0:
        print(
            f"Warning: no pricing for OpenRouter model {spec.model!r}; "
            "cost will be 0 unless the API returns cost.",
            file=sys.stderr,
        )
    return inp, out


_LINE_ITEM_KEYS = frozenset({"description", "price", "quantity", "amount"})


def _looks_like_line_item(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = _dict_key_names(obj)
    return bool(keys & _LINE_ITEM_KEYS) and "line_items" not in keys


def _normalize_line_item_dict(item: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in item.items():
        if key == "issues":
            continue
        lk = key.lower()
        if lk in ("description", "price", "quantity", "amount"):
            out[lk] = val
        elif lk in ("amount_tax_type", "tax_status", "amount_type") and "amount_tax_type" not in out:
            out["amount_tax_type"] = val
    return out


def _dict_key_names(obj: dict[str, Any]) -> set[str]:
    return {k.lower() for k in obj.keys()}


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def normalize_invoice_payload(raw: str) -> str:
    """Coerce common OpenRouter JSON shapes into PaintsInvoice-compatible dict."""
    data: Any = json.loads(_strip_json_fence(raw))
    if isinstance(data, list):
        if not data:
            raise ValueError("Model returned empty JSON array")
        dicts = [x for x in data if isinstance(x, dict)]
        if dicts and all(_looks_like_line_item(x) for x in dicts):
            data = {
                "line_items": [_normalize_line_item_dict(x) for x in dicts],
                "issues": [],
            }
        elif len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
        else:
            data = data[0]
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    if _looks_like_line_item(data) and "line_items" not in data:
        data = {"line_items": [_normalize_line_item_dict(data)], "issues": []}
    if data.get("issues") is None:
        data["issues"] = []
    if "line_items" not in data:
        for key in ("items", "lineItems", "products", "invoice_data", "data", "rows"):
            if isinstance(data.get(key), list):
                data["line_items"] = data.pop(key)
                break
    if isinstance(data.get("line_items"), list):
        data["line_items"] = [
            _normalize_line_item_dict(li) if isinstance(li, dict) else li
            for li in data["line_items"]
        ]
    return json.dumps(data)


def _is_credit_error(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) == 402:
        return True
    text = str(exc)
    return "Error code: 402" in text or "requires more credits" in text


def parse_openrouter_structured(
    client: OpenAI,
    image_path: Path,
    text: str,
    text_format: type[BaseModel],
    model_id: str,
    usage: RunUsage | None = None,
) -> BaseModel:
    mime, data = encode_image(image_path)
    image_url = f"data:{mime};base64,{data}"
    if hasattr(text_format, "model_json_schema"):
        schema = text_format.model_json_schema()
    else:
        schema = text_format.schema()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    def finish(response: object) -> BaseModel:
        if usage is not None:
            usage.add_openrouter_usage(
                getattr(response, "usage", None),
                openrouter_extra_cost(response),
            )
        choice = response.choices[0]
        raw = choice.message.content
        if not raw:
            raise RuntimeError(f"Empty model output for {image_path.name}")
        if text_format is PaintsInvoice:
            cleaned = normalize_invoice_payload(raw)
        else:
            cleaned = _strip_json_fence(raw)
        if hasattr(text_format, "model_validate_json"):
            return text_format.model_validate_json(cleaned)
        return text_format.parse_raw(cleaned)

    primary_format = {
        "type": "json_schema",
        "json_schema": {
            "name": text_format.__name__,
            "schema": schema,
            "strict": True,
        },
    }

    def create(response_format: dict[str, Any]) -> object:
        return client.chat.completions.create(
            model=model_id,
            messages=messages,
            response_format=response_format,
        )

    def primary() -> BaseModel:
        return finish(create(primary_format))

    try:
        return primary()
    except Exception as exc:
        if _is_credit_error(exc):
            raise
        print(
            f"  Retrying {image_path.name} with primary method...",
            file=sys.stderr,
        )
        try:
            return primary()
        except Exception as retry_exc:
            if _is_credit_error(retry_exc):
                raise
        response = create({"type": "json_object"})
        return finish(response)


def retry_extract_gemini(
    client: OpenAI,
    model_id: str,
    image_path: Path,
    invoice: PaintsInvoice,
    usage: RunUsage | None = None,
) -> RetryResult:
    text = (
        f"{RETRY_PROMPT}\n\n"
        f"{mismatch_notes(invoice)}\n\n"
        f"Current extraction:\n{to_json(invoice)}"
    )
    result = parse_openrouter_structured(
        client, image_path, text, RetryResult, model_id, usage
    )
    assert isinstance(result, RetryResult)
    return result


def extract_invoice_gemini(
    client: OpenAI,
    model_id: str,
    image_path: Path,
    usage: RunUsage | None = None,
) -> tuple[PaintsInvoice, list[str], bool, int, list[str]]:
    invoice = parse_openrouter_structured(
        client, image_path, PROMPT, PaintsInvoice, model_id, usage
    )
    assert isinstance(invoice, PaintsInvoice)
    tries = 1
    new_lines: list[str] = []

    while MATH_CHECK_ENABLED and tries < MAX_TRIES and math_is_far(invoice):
        print(
            f"  Retry {tries + 1}/{MAX_TRIES}: price x quantity does not match Amount...",
            file=sys.stderr,
        )
        before = far_line_count(invoice)
        retry = retry_extract_gemini(client, model_id, image_path, invoice, usage)
        invoice = retry.invoice
        tries += 1
        if far_line_count(invoice) < before:
            suggested = normalize_prompt_line(retry.prompt_line)
            if suggested:
                new_lines.append(suggested)
                print(
                    f"  Suggested prompt line for {image_path.name}: {suggested}",
                    file=sys.stderr,
                )

    data = to_dict(invoice)
    model_issues = data.get("issues") or []
    issues = list(dict.fromkeys(list(model_issues) + rule_issues(invoice)))
    math_ok = (not math_is_far(invoice)) if MATH_CHECK_ENABLED else True
    return invoice, issues, math_ok, tries, new_lines


def load_existing_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, list):
        return {"images": data}
    return data


def save_summary(
    path: Path,
    *,
    model_id: str,
    label: str,
    images_folder: Path,
    math_check_enabled: bool,
    results: list[dict[str, Any]],
    run_usage: RunUsage,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": model_id,
        "model_label": label,
        "provider": "openrouter",
        "images_folder": str(images_folder.resolve()),
        "math_check_enabled": math_check_enabled,
        "images_processed": len(results),
        "usage": run_usage.as_dict(),
        "images": results,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Test Run invoices with Gemini (OpenRouter), standalone script."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES,
        help=f"Folder of invoice images (default: {DEFAULT_IMAGES.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=DEFAULT_MODELS_CONFIG,
        help="Model JSON (default: models_gemini.json)",
    )
    parser.add_argument(
        "--no-math-check",
        action="store_true",
        help="Single pass per image (no price×quantity retries)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip source_file names already present in --output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N images (0 = all)",
    )
    args = parser.parse_args()

    set_math_check(not args.no_math_check)

    or_key = os.getenv("OPENROUTER_API_KEY")
    if not or_key:
        print(
            "Set OPENROUTER_API_KEY in .env (this folder or parent P&P LLM Testing).",
            file=sys.stderr,
        )
        return 1

    if not args.models_config.exists():
        print(f"Models config not found: {args.models_config}", file=sys.stderr)
        return 1

    spec = load_model_specs(args.models_config)[0]
    if spec.model != GEMINI_MODEL_ID:
        print(
            f"Warning: expected {GEMINI_MODEL_ID!r}, config has {spec.model!r}.",
            file=sys.stderr,
        )

    images_path = args.images.resolve()
    if not images_path.exists():
        print(f"Images folder not found: {images_path}", file=sys.stderr)
        return 1

    images = collect_images(images_path)
    if not images:
        print(f"No invoice images in {images_path}", file=sys.stderr)
        return 1

    existing = load_existing_results(args.output) if args.resume else {}
    done_names = {
        entry["source_file"]
        for entry in existing.get("images", [])
        if isinstance(entry, dict) and "source_file" in entry and "error" not in entry
    }
    results: list[dict[str, Any]] = list(existing.get("images", [])) if args.resume else []

    inp_rate, out_rate = rates_for_spec(spec)
    run_usage = RunUsage(spec.model, inp_rate, out_rate)
    if args.resume and existing.get("usage"):
        prev = existing["usage"]
        run_usage.calls = int(prev.get("api_calls") or 0)
        run_usage.input_tokens = int(prev.get("input_tokens") or 0)
        run_usage.output_tokens = int(prev.get("output_tokens") or 0)
        if prev.get("cost_from_provider"):
            run_usage.reported_cost_usd = float(prev.get("estimated_cost_usd") or 0)

    client = OpenAI(base_url=OPENROUTER_BASE, api_key=or_key)

    pending = [p for p in images if p.name not in done_names]
    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"Model: {spec.label} ({spec.model})", file=sys.stderr)
    print(f"Images folder: {images_path}", file=sys.stderr)
    print(f"Math check: {'on' if MATH_CHECK_ENABLED else 'off'}", file=sys.stderr)
    print(
        f"To process: {len(pending)} image(s)"
        + (f" (skipped {len(done_names)} from resume)" if done_names else ""),
        file=sys.stderr,
    )

    for image_path in pending:
        print(f"Extracting {image_path.name}...", file=sys.stderr)
        image_usage = RunUsage(spec.model, inp_rate, out_rate)
        entry: dict[str, Any] = {"source_file": image_path.name}
        try:
            invoice, issues, math_ok, tries, new_lines = extract_invoice_gemini(
                client, spec.model, image_path, image_usage
            )
            data = to_dict(invoice)
            data.pop("issues", None)
            entry.update(data)
            entry["math_ok"] = math_ok
            entry["tries"] = tries
            entry["new_prompt_lines"] = new_lines
            entry["issues"] = issues
            entry["usage"] = image_usage.as_dict()
        except (RuntimeError, ValidationError, Exception) as exc:
            entry["error"] = str(exc)
            entry["usage"] = image_usage.as_dict()

        run_usage.calls += image_usage.calls
        run_usage.input_tokens += image_usage.input_tokens
        run_usage.output_tokens += image_usage.output_tokens
        if image_usage.reported_cost_usd is not None:
            run_usage.reported_cost_usd = (run_usage.reported_cost_usd or 0.0) + (
                image_usage.reported_cost_usd
            )

        if args.resume:
            results = [r for r in results if r.get("source_file") != image_path.name]
        results.append(entry)

        save_summary(
            args.output,
            model_id=spec.model,
            label=spec.label,
            images_folder=images_path,
            math_check_enabled=MATH_CHECK_ENABLED,
            results=results,
            run_usage=run_usage,
        )

    print(f"Saved {args.output.resolve()}", file=sys.stderr)
    errors = sum(1 for r in results if "error" in r)
    print(
        f"Done: {len(results)} total row(s), {errors} error(s), "
        f"cost ~${run_usage.estimated_cost_usd():.4f}",
        file=sys.stderr,
    )
    return 0 if errors == 0 or len(pending) > errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
