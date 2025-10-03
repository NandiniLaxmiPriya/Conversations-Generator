# save as: pages_by_type.py
from datasets import load_from_disk
import pandas as pd
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Count pages by PDF type.")
    parser.add_argument("--data_dir", required=True, help="Path to DocLayNet subset (save_to_disk folder)")
    parser.add_argument("--out_dir", default="analysis_out", help="Where to write CSVs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(args.data_dir)

    # Extract minimal metadata needed
    rows = []
    for row in ds:
        m = row.get("metadata", {}) or {}
        rows.append({
            "original_filename": m.get("original_filename"),
            "collection": m.get("collection"),
            "doc_category": m.get("doc_category"),
            "page_no": m.get("page_no"),
            "num_pages": m.get("num_pages"),
        })
    df = pd.DataFrame(rows)

    # ---- 1) Pages per collection (TYPE = collection)
    pages_per_collection = (
        df.groupby("collection")
          .size()
          .reset_index(name="pages_in_subset")
          .sort_values("pages_in_subset", ascending=False)
    )
    pages_per_collection.to_csv(out_dir / "pages_per_collection.csv", index=False)

    # ---- 2) Pages per doc_category (TYPE = doc_category), if available
    if df["doc_category"].notna().any():
        pages_per_doc_category = (
            df.groupby("doc_category")
              .size()
              .reset_index(name="pages_in_subset")
              .sort_values("pages_in_subset", ascending=False)
        )
        pages_per_doc_category.to_csv(out_dir / "pages_per_doc_category.csv", index=False)
    else:
        pages_per_doc_category = None

    # ---- 3) Pages per PDF (TYPE = original_filename)
    pages_per_pdf = (
        df.groupby(["collection", "doc_category", "original_filename"])
          .size()
          .reset_index(name="pages_in_subset")
          .sort_values("pages_in_subset", ascending=False)
    )
    # Add full document page count (should be constant per PDF; use max for safety)
    full_pages = (
        df.groupby("original_filename")["num_pages"]
          .max()
          .rename("num_pages_full")
          .reset_index()
    )
    pages_per_pdf = pages_per_pdf.merge(full_pages, on="original_filename", how="left")
    pages_per_pdf["coverage_pct"] = (pages_per_pdf["pages_in_subset"] / pages_per_pdf["num_pages_full"]) * 100.0
    pages_per_pdf.to_csv(out_dir / "pages_per_pdf.csv", index=False)

    # ---- 4) Optional combined view: per collection, how many PDFs and avg pages/PDF in your subset
    unique_pdfs = (
        pages_per_pdf.groupby("collection")["original_filename"]
        .nunique()
        .reset_index(name="unique_pdfs")
    )
    combined = pages_per_collection.merge(unique_pdfs, on="collection", how="left")
    combined["avg_pages_per_pdf_in_subset"] = combined["pages_in_subset"] / combined["unique_pdfs"].clip(lower=1)
    combined = combined.sort_values("pages_in_subset", ascending=False)
    combined.to_csv(out_dir / "pages_per_collection_with_pdf_stats.csv", index=False)

    # ---- Console summary
    print("\n=== Pages per collection (top 10) ===")
    print(pages_per_collection.head(10).to_string(index=False))

    if pages_per_doc_category is not None:
        print("\n=== Pages per doc_category (top 10) ===")
        print(pages_per_doc_category.head(10).to_string(index=False))
    else:
        print("\n(doc_category not available or all missing in this subset)")

    print("\n=== Top PDFs by pages_in_subset (top 10) ===")
    cols = ["original_filename", "collection", "doc_category", "pages_in_subset", "num_pages_full", "coverage_pct"]
    print(pages_per_pdf[cols].head(10).to_string(index=False))

    print(f"\nWrote CSVs to: {out_dir.resolve()}")
    print(" - pages_per_collection.csv")
    if pages_per_doc_category is not None:
        print(" - pages_per_doc_category.csv")
    print(" - pages_per_pdf.csv")
    print(" - pages_per_collection_with_pdf_stats.csv")

if __name__ == "__main__":
    main()
