from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from PIL import Image
import os

def preprocess_image(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Denoising (optional)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Adaptive thresholding works better for Indian scripts
    processed = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return processed


def extract_text_from_pdf(pdf_filepath, output_filepath,pdfType):
    try:
        images = convert_from_path(pdf_filepath, dpi=350)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    with open(output_filepath, "w", encoding="utf-8") as outfile:
        for i, image in enumerate(images):
            print(f"Processing page {i + 1}...")
            processed_image = preprocess_image(image)
            language = ""
            if pdfType == "1":
                language = "tel"
            elif pdfType == "2":
                language = "eng"
            else:
                print(f"Warning: Invalid pdfinputType '{pdfType}'. Defaulting to English.")
                language = "eng"
            try:
                text = pytesseract.image_to_string(processed_image, lang="eng")
                outfile.write(text)
            except Exception as e:
                print(f"Error processing page {i + 1}: {e}")
                outfile.write(f"--- Page {i + 1} ---\n[Error processing text]\n\n")

    print(f"Text extraction complete. Output saved to {output_filepath}")

def pdfinputType():
    pdfType = input("Enter 1.Telugu Text PDF 2.English Text PDF") 
    if pdfType=="1":
        print("1. Telugu")
        input_file = "engtel.pdf"
        output_file = "Telugu_extracted_text_tesseract.txt"
        if not os.path.exists(input_file):
            print(f"PDF file '{input_file}' not found.")
        else:
            extract_text_from_pdf(input_file, output_file,pdfType)
    else:
        print("2. English")
        input_file = "Photosynth_input.pdf"
        output_file = "English_extracted_text_tesseract.txt"
        if not os.path.exists(input_file):
            print(f"PDF file '{input_file}' not found.")
        else:
            extract_text_from_pdf(input_file, output_file, pdfType)
if __name__ == "__main__":
    pdfinputType()