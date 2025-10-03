import pandas as pd
from datasets import load_from_disk

# # DocLayNet category mapping
# id2label = {
#     1: "Caption",
#     2: "Footnote",
#     3: "Formula",
#     4: "List-item",
#     5: "Page-footer",
#     6: "Page-header",
#     7: "Picture",
#     8: "Section-header",
#     9: "Table",
#     10: "Text",
#     11: "Title"
# }

# # Load your category_distribution.csv
# cat_df = pd.read_csv("analysis_out/category_distribution.csv")

# # Add human-readable label
# cat_df["label"] = cat_df["category_id"].map(id2label)

# # Save as new file
# cat_df.to_csv("analysis_out/category_distribution_labeled.csv", index=False)

# print(cat_df)

# Load your 10k dataset
ds = load_from_disk("DocLayNet-10k")

rows = []
for row in ds:
    meta = row["metadata"]
    rows.append({
        "original_filename": meta.get("original_filename"),
        "collection": meta.get("collection"),
        "doc_category": meta.get("doc_category"),
        "page_no": meta.get("page_no"),
        "num_pages": meta.get("num_pages"),
    })

df = pd.DataFrame(rows)

# 1. How many unique PDFs in this subset
unique_pdfs = df["original_filename"].nunique()
print(f"Unique PDFs in 10k subset: {unique_pdfs}")

# 2. Page count by collection type
collection_counts = df["collection"].value_counts()
print("\nPages per collection:")
print(collection_counts)

# 3. Number of unique PDFs per collection
pdfs_per_collection = df.groupby("collection")["original_filename"].nunique()
print("\nUnique PDFs per collection:")
print(pdfs_per_collection)

# 4. Top PDFs with most pages in the subset
pages_per_pdf = df["original_filename"].value_counts().head(10)
print("\nTop 10 PDFs by number of pages in this subset:")
print(pages_per_pdf)

# 5. Optional: save summary CSV
df_summary = df.groupby(["collection", "original_filename"]).size().reset_index(name="pages_in_subset")
df_summary.to_csv("analysis_out/documents_summary.csv", index=False)
print("\nSaved per-document summary: analysis_out/documents_summary.csv")
