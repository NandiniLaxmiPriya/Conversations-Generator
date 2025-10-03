# save as: data_main.py
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

# Optional (only used if you request --sample_vis > 0)
try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except Exception:
    PIL_OK = False


# ---------------------------
# Helpers
# ---------------------------

def try_get_metadata_dim(meta: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Prefer coco_width/height; fallback to original_* if present."""
    w = meta.get("coco_width") or meta.get("original_width")
    h = meta.get("coco_height") or meta.get("original_height")
    if w is None or h is None:
        return (None, None)
    try:
        return (int(round(float(w))), int(round(float(h))))
    except Exception:
        return (None, None)


def bbox_area_frac(b: List[float], img_w: Optional[int], img_h: Optional[int]) -> float:
    # Expecting COCO-style [x, y, w, h]
    if not img_w or not img_h or img_w <= 0 or img_h <= 0:
        return float("nan")
    return (float(b[2]) * float(b[3])) / float(img_w * img_h)


def bbox_aspect_ratio(b: List[float]) -> float:
    # w/h, but guard divide-by-zero
    h = float(b[3])
    return (float(b[2]) / h) if h != 0 else float("nan")


def iter_pdf_cell_dicts(pdf_cells: Any) -> Iterable[Dict[str, Any]]:
    """
    Yields dict-like text cells from DocLayNet's pdf_cells field, which can be:
      - List[Dict]
      - List[List[Dict]]
      - None / []
    Any non-dict items are skipped safely.
    """
    if not pdf_cells:
        return
    if isinstance(pdf_cells, list):
        for item in pdf_cells:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        yield sub
            # else: ignore weird types silently


def count_seg_items(segs_obj: Any) -> int:
    """
    Robustly count segmentation entries across possible shapes:
      - List[List[List[float]]]  (common polygon format)
      - List[List[float]]        (already polygon)
      - List[...]                (fallback)
    """
    if not segs_obj or not isinstance(segs_obj, list):
        return 0
    count = 0
    for item in segs_obj:
        if isinstance(item, list):
            count += 1
        else:
            count += 1
    return count


# ---------------------------
# Core per-row analysis
# ---------------------------

def analyze_sample(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = row.get("metadata", {}) or {}
    cw, ch = try_get_metadata_dim(meta)

    # counts (guard for None)
    bboxes = row.get("bboxes") or []
    segs   = row.get("segmentation") or []
    pdfc   = row.get("pdf_cells") or []

    # category ids: normalize to list[int]
    cat_ids = row.get("category_id", []) or []
    if isinstance(cat_ids, (int, np.integer)):
        cat_ids = [int(cat_ids)]
    elif isinstance(cat_ids, list):
        cat_ids = [int(x) for x in cat_ids if x is not None]
    else:
        cat_ids = []

    # bbox metrics
    areas, ars, widths, heights = [], [], [], []
    for b in bboxes:
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        areas.append(bbox_area_frac(b, cw, ch))
        ars.append(bbox_aspect_ratio(b))
        widths.append(float(b[2]))
        heights.append(float(b[3]))

    # text cell metrics (flatten nested structures)
    cell_chars, cell_words = [], []
    n_cells = 0
    for cell in iter_pdf_cell_dicts(pdfc):
        txt = (cell.get("text") or "").strip()
        if txt:
            cell_chars.append(len(txt))
            cell_words.append(len(txt.split()))
        n_cells += 1

    return {
        "image_width": cw,
        "image_height": ch,
        "n_bboxes": int(len(bboxes)),
        "n_segmentations": int(count_seg_items(segs)),
        "n_pdf_cells": int(n_cells),
        "bbox_area_frac_mean": float(np.nanmean(areas)) if areas else float("nan"),
        "bbox_area_frac_median": float(np.nanmedian(areas)) if areas else float("nan"),
        "bbox_aspect_ratio_mean": float(np.nanmean(ars)) if ars else float("nan"),
        "bbox_aspect_ratio_median": float(np.nanmedian(ars)) if ars else float("nan"),
        "bbox_width_mean": float(np.mean(widths)) if widths else float("nan"),
        "bbox_height_mean": float(np.mean(heights)) if heights else float("nan"),
        "cell_chars_mean": float(np.mean(cell_chars)) if cell_chars else float("nan"),
        "cell_words_mean": float(np.mean(cell_words)) if cell_words else float("nan"),
        "category_ids": cat_ids,
    }


# ---------------------------
# Visualization
# ---------------------------

def visualize_samples(ds, out_dir: Path, n: int = 12) -> None:
    """
    Draw bboxes over a few samples and save PNGs.
    Only uses the first N examples; adjust as needed.
    """
    if not PIL_OK:
        print("[!] PIL not available. Skipping sample visualizations.")
        return
    vis_dir = out_dir / "sample_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    n = min(max(int(n), 0), len(ds))
    for i in tqdm(range(n), desc="Visualizing"):
        row = ds[i]
        # Load PIL image (datasets usually decodes to PIL Image)
        try:
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
        except Exception:
            continue

        draw = ImageDraw.Draw(img)
        bboxes = row.get("bboxes", []) or []
        cats   = row.get("category_id", []) or []
        if isinstance(cats, (int, np.integer)):
            cats = [int(cats)]

        for j, b in enumerate(bboxes):
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue
            x, y, w, h = b
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
            # Label with category id if aligned
            if j < len(cats) and cats[j] is not None:
                draw.text((x + 2, y + 2), str(int(cats[j])))

        img.save(vis_dir / f"sample_{i:04d}.png")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze a local DocLayNet subset saved with save_to_disk.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to folder created by save_to_disk (e.g., DocLayNet-10k)")
    parser.add_argument("--out_dir", type=str, default="analysis_out",
                        help="Output directory for CSVs/plots")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: limit number of rows to analyze for speed")
    parser.add_argument("--sample_vis", type=int, default=0,
                        help="Draw bboxes for N samples (requires Pillow). 0 disables.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[i] Loading dataset from {data_dir.resolve()} ...")
    ds = load_from_disk(str(data_dir))

    if isinstance(args.limit, int) and args.limit > 0:
        n = min(args.limit, len(ds))
        print(f"[i] Limiting analysis to first {n} rows")
        ds = ds.select(range(n))

    # Basic summary
    n_rows = len(ds)
    fields = list(ds.features.keys())
    print(f"[✔] Loaded {n_rows} rows")
    print(f"[i] Fields: {fields}")

    # Per-row analysis + collect per-page metadata for doc/collection summaries
    rows_stats: List[Dict[str, Any]] = []
    cat_counter: Counter = Counter()
    img_w: List[int] = []
    img_h: List[int] = []

    # Collect per-page metadata for document/collection grouping
    meta_rows: List[Dict[str, Any]] = []

    for i in tqdm(range(n_rows), desc="Analyzing"):
        row = ds[i]
        st = analyze_sample(row)
        rows_stats.append(st)

        # Collect image dims
        if st["image_width"] and st["image_height"]:
            img_w.append(int(st["image_width"]))
            img_h.append(int(st["image_height"]))

        # Category counts
        cat_counter.update(st["category_ids"])

        # Metadata for doc/collection summaries
        m = row.get("metadata", {}) or {}
        meta_rows.append({
            "original_filename": m.get("original_filename"),
            "collection": m.get("collection"),
            "doc_category": m.get("doc_category"),
            "page_no": m.get("page_no"),
            "num_pages": m.get("num_pages"),
        })

    # DataFrames
    df = pd.DataFrame(rows_stats)
    df.to_csv(out_dir / "per_page_stats.csv", index=False)

    cat_total = max(1, sum(cat_counter.values()))
    cat_df = pd.DataFrame(
        [{"category_id": k, "count": v, "pct": (v / cat_total) * 100.0}
         for k, v in sorted(cat_counter.items())]
    )
    cat_df.to_csv(out_dir / "category_distribution.csv", index=False)

    # Aggregate metrics
    summary = {
        "n_rows": n_rows,
        "fields": fields,
        "n_unique_category_ids": int(len(cat_counter)),
        "top_categories": cat_counter.most_common(10),
        "image_width_mean": float(np.mean(img_w)) if img_w else None,
        "image_height_mean": float(np.mean(img_h)) if img_h else None,
        "bbox_count_mean": float(df["n_bboxes"].mean()) if len(df) else None,
        "bbox_count_median": float(df["n_bboxes"].median()) if len(df) else None,
        "bbox_area_frac_mean_of_means": float(df["bbox_area_frac_mean"].dropna().mean())
            if df["bbox_area_frac_mean"].notna().any() else None,
        "bbox_aspect_ratio_mean_of_means": float(df["bbox_aspect_ratio_mean"].dropna().mean())
            if df["bbox_aspect_ratio_mean"].notna().any() else None,
        "pdf_cells_count_mean": float(df["n_pdf_cells"].mean()) if len(df) else None,
        "pdf_cell_chars_mean_of_means": float(df["cell_chars_mean"].dropna().mean())
            if df["cell_chars_mean"].notna().any() else None,
        "pdf_cell_words_mean_of_means": float(df["cell_words_mean"].dropna().mean())
            if df["cell_words_mean"].notna().any() else None,
        "segmentation_presence_pct": float((df["n_segmentations"] > 0).mean() * 100.0)
            if len(df) else None,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Quick, human-readable printout
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== Category distribution (top 10) ===")
    for cat, cnt in cat_counter.most_common(10):
        print(f"category_id={cat} -> {cnt}")

    # Optional: a few sample visualizations
    if args.sample_vis and args.sample_vis > 0:
        if not PIL_OK:
            print("[!] PIL is not installed; run: pip install pillow")
        else:
            print(f"[i] Creating {args.sample_vis} sample visualizations...")
            visualize_samples(ds, out_dir, n=int(args.sample_vis))

    # Extra: aggregate bbox geometry to CSV for deeper EDA
    print("[i] Flattening per-bbox geometry for EDA CSV...")
    flat = {
        "page_index": [],
        "category_id": [],
        "bbox_x": [],
        "bbox_y": [],
        "bbox_w": [],
        "bbox_h": [],
        "area_frac": [],
        "aspect_ratio": [],
    }
    for i in tqdm(range(n_rows), desc="Flattening"):
        row = ds[i]
        meta = row.get("metadata", {}) or {}
        cw, ch = try_get_metadata_dim(meta)
        cats = row.get("category_id", []) or []
        if isinstance(cats, (int, np.integer)):
            cats = [int(cats)]
        elif isinstance(cats, list):
            cats = [int(x) for x in cats if x is not None]
        else:
            cats = []
        bboxes = row.get("bboxes", []) or []
        for j, b in enumerate(bboxes):
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue
            cat = int(cats[j]) if j < len(cats) else None
            ar = bbox_aspect_ratio(b)
            af = bbox_area_frac(b, cw, ch)
            flat["page_index"].append(i)
            flat["category_id"].append(cat)
            flat["bbox_x"].append(float(b[0]))
            flat["bbox_y"].append(float(b[1]))
            flat["bbox_w"].append(float(b[2]))
            flat["bbox_h"].append(float(b[3]))
            flat["area_frac"].append(af)
            flat["aspect_ratio"].append(ar)

    pd.DataFrame(flat).to_csv(out_dir / "bbox_geometry.csv", index=False)

    # ===========================
    # NEW: Document & Collection summaries
    # ===========================
    print("[i] Building document and collection summaries...")
    meta_df = pd.DataFrame(meta_rows)

    # Pages per collection (count of rows)
    pages_per_collection = (
        meta_df.groupby("collection")
        .size()
        .reset_index(name="pages_in_subset")
        .sort_values("pages_in_subset", ascending=False)
    )
    pages_per_collection.to_csv(out_dir / "pages_per_collection.csv", index=False)

    # Unique PDFs per collection
    unique_pdfs_per_collection = (
        meta_df.groupby("collection")["original_filename"]
        .nunique()
        .reset_index(name="unique_pdfs")
        .sort_values("unique_pdfs", ascending=False)
    )
    unique_pdfs_per_collection.to_csv(out_dir / "unique_pdfs_per_collection.csv", index=False)

    # Per-PDF summary (pages present in subset, and coverage vs total document pages)
    docs = (
        meta_df.groupby(["collection", "doc_category", "original_filename"])
        .agg(
            pages_in_subset=("page_no", "count"),
            num_pages_full=("num_pages", "max")  # should be constant per PDF; max for safety
        )
        .reset_index()
    )
    # coverage = pages_in_subset / num_pages_full
    docs["coverage_pct"] = (
        np.where(docs["num_pages_full"] > 0,
                 (docs["pages_in_subset"] / docs["num_pages_full"]) * 100.0,
                 np.nan)
    )
    docs.sort_values(["pages_in_subset", "coverage_pct"], ascending=[False, False], inplace=True)
    docs.to_csv(out_dir / "documents_summary.csv", index=False)

    # Console quick stats
    print(f"\nUnique PDFs in subset: {meta_df['original_filename'].nunique()}")
    print("\nPages per collection (top 10):")
    print(pages_per_collection.head(10).to_string(index=False))
    print("\nUnique PDFs per collection (top 10):")
    print(unique_pdfs_per_collection.head(10).to_string(index=False))
    print("\nTop 10 PDFs by pages_in_subset:")
    print(docs[["original_filename", "collection", "pages_in_subset", "num_pages_full", "coverage_pct"]]
          .head(10).to_string(index=False))

    print(f"\n[✔] Analysis complete. Files saved under: {out_dir.resolve()}")
    print("    - per_page_stats.csv")
    print("    - category_distribution.csv")
    print("    - bbox_geometry.csv")
    print("    - pages_per_collection.csv")
    print("    - unique_pdfs_per_collection.csv")
    print("    - documents_summary.csv")
    print("    - summary.json")
    if args.sample_vis and PIL_OK:
        print("    - sample_visualizations/*.png")


if __name__ == "__main__":
    main()
