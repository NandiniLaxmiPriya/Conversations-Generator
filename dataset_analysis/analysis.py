# save as: download_doclaynet_10k.py
import os
import argparse
from datasets import load_dataset, Dataset
from pathlib import Path

def load_first_n(dataset_name: str, split: str, n: int, cache_dir: str | None):
    """
    Load the first n rows from a split using HF slicing so that only the needed
    files are fetched.
    """
    return load_dataset(
        dataset_name,
        split=f"{split}[:{n}]",
        cache_dir=cache_dir,
    )

def main():
    parser = argparse.ArgumentParser(description="Download a 10k subset of DocLayNet locally.")
    parser.add_argument("--limit", type=int, default=10000, help="Number of samples to fetch")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (usually 'train')")
    parser.add_argument("--out_dir", type=str, default="DocLayNet-10k", help="Output directory for save_to_disk")
    parser.add_argument("--cache_dir", type=str, default=None, help="HF datasets cache dir (optional)")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    if out_path.exists() and (out_path / "state.json").exists():
        print(f"[✔] Found existing subset at: {out_path.resolve()}")
        print("    Skipping download. If you want to re-download, delete the folder and rerun.")
        return

    # Try v1.2 first, then fall back to v1.1
    candidates = ["ds4sd/DocLayNet-v1.2", "ds4sd/DocLayNet-v1.1"]

    last_err = None
    ds = None
    for name in candidates:
        try:
            print(f"[i] Attempting to load {name} split '{args.split}' with first {args.limit} rows...")
            ds = load_first_n(name, args.split, args.limit, args.cache_dir)
            print(f"[✔] Loaded: {name} ({len(ds)} rows)")
            break
        except Exception as e:
            print(f"[!] Failed with {name}: {e}")
            last_err = e

    if ds is None:
        raise SystemExit(f"Could not load DocLayNet (tried {candidates}). Last error:\n{last_err}")

    # Save the subset locally (this also records feature schema)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"[i] Saving subset to: {out_path.resolve()}")
    ds.save_to_disk(str(out_path))
    print("[✔] Done.")

    # Show a tiny peek
    print("\n[peek] Features:", ds.features)
    print("[peek] First row keys:", list(ds[0].keys()))
    print("[peek] Example row index 0:", {k: type(ds[0][k]).__name__ for k in ds[0].keys()})

if __name__ == "__main__":
    # Optional: speed up hf-hub transfers (if installed)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    main()
