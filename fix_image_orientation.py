"""
Detect document orientation with PaddleOCR (PP-LCNet_x1_0_doc_ori) and save upright copies.

Orientation is predicted only by the Paddle model (no EXIF). If the first-pass
score is below 0.8, the original is copied into a low_confidence/ subfolder
under the destination, then the script tries all four corrections (0/90/180/270)
and saves the result in that subfolder.

Install (once):
    pip install paddleocr

Examples:
    python fix_image_orientation.py photo.jpg -o corrected/
    python fix_image_orientation.py ./invoices -o ./invoices_upright --recursive
    python fix_image_orientation.py photo.jpg --detect-only
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

# Faster startup; model still downloads on first run.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

MODEL_NAME = "PP-LCNet_x1_0_doc_ori"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CANDIDATE_ANGLES = ("0", "90", "180", "270")
DEFAULT_FALLBACK_BELOW = 0.8
LOW_CONFIDENCE_SUBDIR = "low_confidence"


@dataclass(frozen=True)
class OrientationResult:
    path: Path
    angle: str
    score: float
    source: str = "primary"
    primary_angle: str | None = None
    primary_score: float | None = None
    saved_to: Path | None = None

    @property
    def needs_rotation(self) -> bool:
        return self.angle != "0"


def _first_scalar(value):
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return value


def _parse_payload(payload) -> tuple[str, float]:
    angle = str(_first_scalar(payload["label_names"]))
    score = float(_first_scalar(payload["scores"]))
    return angle, score


def read_bgr(image_path: Path):
    """Read BGR image; handles Windows paths OpenCV cannot open directly."""
    image_path = Path(image_path)
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    # Fallback for unusual layouts.
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def write_bgr(image_path: Path, image) -> None:
    image_path = Path(image_path)
    suffix = image_path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image for: {image_path}")
    encoded.tofile(str(image_path))


def rotate_upright_bgr(image, angle: str):
    """Same rotation math as PaddleX doc preprocessor (warpAffine, expand canvas)."""
    degrees = int(angle)
    if degrees < 0 or degrees >= 360:
        raise ValueError(f"Unexpected orientation label from Paddle: {angle!r}")
    if degrees < 1:
        return image

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width - width) / 2
    matrix[1, 2] += (new_height - height) / 2
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
    )


def deskew_bgr(image, max_angle: float = 12.0):
    """Small in-plane tilt correction for photos (not handled by doc orientation model)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.3 or abs(angle) > max_angle:
        return image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


class DocumentOrientationFixer:
    def __init__(self, device: str | None = None):
        from paddleocr import DocImgOrientationClassification

        self._model = DocImgOrientationClassification(
            model_name=MODEL_NAME,
            device=device,
        )

    def _predict_path(self, image_path: Path):
        outputs = self._model.predict(str(image_path), batch_size=1)
        return outputs[0]

    def _predict_bgr(self, image_bgr):
        outputs = self._model.predict(image_bgr, batch_size=1)
        return outputs[0]

    def _classify(self, image_path: Path) -> tuple[str, float]:
        paddle_result = self._predict_path(image_path)
        payload = paddle_result.json.get("res", paddle_result.json)
        return _parse_payload(payload)

    def _classify_bgr(self, image_bgr) -> tuple[str, float]:
        paddle_result = self._predict_bgr(image_bgr)
        payload = paddle_result.json.get("res", paddle_result.json)
        return _parse_payload(payload)

    def _primary_classify(self, image_path: Path) -> tuple[str, float]:
        image = read_bgr(image_path)
        if image is None:
            return self._classify(image_path)
        return self._classify_bgr(image)

    def detect(
        self,
        image_path: str | Path,
        *,
        fallback_below: float = DEFAULT_FALLBACK_BELOW,
    ) -> OrientationResult:
        """Use primary Paddle prediction; vote across 4 rotations only if score < fallback_below."""
        image_path = Path(image_path)
        image = read_bgr(image_path)
        if image is None:
            angle, score = self._classify(image_path)
            return OrientationResult(path=image_path, angle=angle, score=score, source="primary")

        primary_angle, primary_score = self._classify_bgr(image)
        if primary_score >= fallback_below:
            return OrientationResult(
                path=image_path,
                angle=primary_angle,
                score=primary_score,
                source="primary",
            )

        best_angle = "0"
        best_upright_score = -1.0
        for candidate in CANDIDATE_ANGLES:
            trial = image if candidate == "0" else rotate_upright_bgr(image, candidate)
            pred_angle, pred_score = self._classify_bgr(trial)
            if pred_angle == "0" and pred_score > best_upright_score:
                best_upright_score = pred_score
                best_angle = candidate

        if best_upright_score >= 0:
            source = "vote" if best_angle != primary_angle else "primary"
            return OrientationResult(
                path=image_path,
                angle=best_angle,
                score=best_upright_score,
                source=source,
                primary_angle=primary_angle,
                primary_score=primary_score,
            )

        return OrientationResult(
            path=image_path,
            angle=primary_angle,
            score=primary_score,
            source="primary",
            primary_angle=primary_angle,
            primary_score=primary_score,
        )

    def fix_file(
        self,
        image_path: str | Path,
        output_path: str | Path,
        *,
        min_confidence: float = 0.0,
        skip_if_upright: bool = True,
        deskew: bool = False,
        fallback_below: float = DEFAULT_FALLBACK_BELOW,
        low_confidence_subdir: str = LOW_CONFIDENCE_SUBDIR,
    ) -> OrientationResult:
        image_path = Path(image_path)
        output_path = Path(output_path)
        primary_angle, primary_score = self._primary_classify(image_path)
        use_fallback = primary_score < fallback_below

        if use_fallback:
            output_path = output_path.parent / low_confidence_subdir / output_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, output_path)
            detect_path = output_path
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            detect_path = image_path

        result = self.detect(detect_path, fallback_below=fallback_below)

        should_rotate = (
            result.needs_rotation
            and result.score >= min_confidence
            and not (skip_if_upright and not result.needs_rotation)
        )

        image = read_bgr(detect_path)
        if image is None:
            raise ValueError(
                f"Could not read image bytes for rotation: {detect_path}. "
                "Paddle may still classify it; check that the file is a valid image."
            )

        if should_rotate:
            image = rotate_upright_bgr(image, result.angle)
        if deskew:
            image = deskew_bgr(image)

        if not should_rotate and not deskew and not use_fallback:
            if image_path.resolve() != output_path.resolve():
                shutil.copy2(image_path, output_path)
            return replace(result, saved_to=output_path)

        write_bgr(output_path, image)
        return replace(result, saved_to=output_path)


def iter_images(root: Path, recursive: bool) -> Iterator[Path]:
    if root.is_file():
        if root.suffix.lower() in IMAGE_SUFFIXES:
            yield root
        return
    pattern = "**/*" if recursive else "*"
    for path in sorted(root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def default_output_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / input_path.name


def resolve_single_output(input_file: Path, output: Path | None) -> Path:
    if output is None:
        return input_file.with_name(f"{input_file.stem}_upright{input_file.suffix}")
    if output.suffix.lower() in IMAGE_SUFFIXES:
        return output
    return output / input_file.name


def write_report(rows: Iterable[dict], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input",
        "output",
        "angle",
        "score",
        "source",
        "primary_angle",
        "primary_score",
        "rotated",
        "low_confidence_folder",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use Paddle PP-LCNet_x1_0_doc_ori to detect document orientation "
            "(0/90/180/270) and save upright images."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Image file or folder containing images",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file (single input) or output folder (directory input)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a folder, include subfolders",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Inference device, e.g. "cpu", "gpu", "gpu:0" (default: auto)',
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="If prediction score is below this, copy the original without rotating",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Print Paddle-predicted angle/score only; do not write images",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional CSV path for batch results",
    )
    parser.add_argument(
        "--deskew",
        action="store_true",
        help="After Paddle rotation, correct small photo tilt (a few degrees)",
    )
    parser.add_argument(
        "--fallback-below",
        type=float,
        default=DEFAULT_FALLBACK_BELOW,
        metavar="SCORE",
        help=(
            "Run 4-way rotation vote only when the first-pass Paddle score is "
            f"below this (default: {DEFAULT_FALLBACK_BELOW})"
        ),
    )
    parser.add_argument(
        "--low-confidence-subdir",
        default=LOW_CONFIDENCE_SUBDIR,
        help=(
            "Subfolder under the destination where sub-threshold images are copied "
            f"before fallback runs (default: {LOW_CONFIDENCE_SUBDIR})"
        ),
    )
    return parser


def format_result_line(result: OrientationResult) -> str:
    parts = [f"angle={result.angle}", f"score={result.score:.4f}", f"source={result.source}"]
    if result.primary_angle is not None:
        parts.append(f"primary={result.primary_angle}@{result.primary_score:.4f}")
    return " ".join(parts)


def main() -> None:
    args = build_parser().parse_args()
    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    fixer = DocumentOrientationFixer(device=args.device)
    detect_kwargs = {"fallback_below": args.fallback_below}
    fix_kwargs = {
        **detect_kwargs,
        "min_confidence": args.min_confidence,
        "deskew": args.deskew,
        "low_confidence_subdir": args.low_confidence_subdir,
    }
    images = list(iter_images(input_path, args.recursive))
    if not images:
        if input_path.is_file():
            raise SystemExit(
                f"Unsupported image type: {input_path.suffix!r}. "
                f"Supported: {', '.join(sorted(IMAGE_SUFFIXES))}"
            )
        raise SystemExit(
            f"No supported image files found under {input_path}. "
            f"Supported: {', '.join(sorted(IMAGE_SUFFIXES))}. "
            "Use --recursive if images are in subfolders."
        )

    if args.detect_only:
        for image in images:
            result = fixer.detect(image, **detect_kwargs)
            print(f"{image}\t{format_result_line(result)}")
        return

    if len(images) == 1 and input_path.is_file():
        out = resolve_single_output(input_path, args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        result = fixer.fix_file(images[0], out, **fix_kwargs)
        saved = result.saved_to or out
        rotated = result.needs_rotation and result.score >= args.min_confidence
        print(
            f"{images[0]} -> {saved} | {format_result_line(result)} "
            f"rotated={'yes' if rotated else 'no'}"
        )
        return

    output_dir = args.output or input_path.parent / f"{input_path.name}_upright"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    for image in images:
        if input_path.is_dir():
            rel = image.relative_to(input_path)
            out_path = output_dir / rel
        else:
            out_path = default_output_path(image, output_dir)
        result = fixer.fix_file(image, out_path, **fix_kwargs)
        saved = result.saved_to or out_path
        rotated = result.needs_rotation and result.score >= args.min_confidence
        low_conf = (
            result.primary_score is not None
            and result.primary_score < args.fallback_below
        )
        print(f"{image} -> {saved} | {format_result_line(result)} rotated={'yes' if rotated else 'no'}")
        report_rows.append(
            {
                "input": str(image),
                "output": str(saved),
                "angle": result.angle,
                "score": f"{result.score:.6f}",
                "source": result.source,
                "primary_angle": result.primary_angle or "",
                "primary_score": (
                    f"{result.primary_score:.6f}" if result.primary_score is not None else ""
                ),
                "rotated": "yes" if rotated else "no",
                "low_confidence_folder": "yes" if low_conf else "no",
            }
        )

    if args.report:
        write_report(report_rows, args.report)


if __name__ == "__main__":
    main()
