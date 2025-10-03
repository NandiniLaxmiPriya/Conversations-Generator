# def extract_text_from_pdf(pdf_filepath, output_filepath):
#     try:
#         doc = fitz.open(pdf_filepath)
#         with open(output_filepath, "w", encoding="utf-8") as outfile:
#             for page in doc:
#                 text = page.get_text("text")  # use default text mode
#                 outfile.write(text + "\n\n")
#         print(f"✅ Successfully extracted Telugu text from '{pdf_filepath}' to '{output_filepath}'")
#     except FileNotFoundError:
#         print(f"❌ File not found: '{pdf_filepath}'")
#     except Exception as e:
#         print(f"❌ Error: {e}")
import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path, output_dir, merged_text_file="merged_text_output.txt"):
    os.makedirs(output_dir, exist_ok=True)

    # Open the PDF
    doc = fitz.open(pdf_path)

    merged_text_path = os.path.join(output_dir, merged_text_file)
    with open(merged_text_path, "w", encoding="utf-8") as merged_file:
        for page_num, page in enumerate(doc, start=1):
            print(f"\nProcessing page {page_num}...")

            # Extract text and append to merged file
            text = page.get_text("text")
            merged_file.write(f"--- Page {page_num} ---\n{text}\n\n")

            # Extract and save images
            image_list = page.get_images(full=True)
            print(f"Found {len(image_list)} images on page {page_num}")

            for img_index, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = os.path.join(output_dir, f"page_{page_num}_img_{img_index}.{image_ext}")

                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)
                print(f"Saved image to {image_filename}")

    doc.close()
    print(f"\n✅ Finished extracting. Combined text saved to '{merged_text_path}'")

if __name__ == "__main__":
    pdf_file = "Photosynth_input.pdf"
    output_folder = "output2_files"
    extract_text_from_pdf(pdf_file, output_folder)
