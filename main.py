import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import base64
import fitz
from langdetect import detect
load_dotenv()
prompt = """ You are an efficient and precise text-based PDF extractor.
You will receive the content of PDF, which may contain English text. \n
Your task is to extract the complete textual content exactly as it appears in the PDF, preserving the original language of each segment.\n
Do not extract text from images, tables and figures.\n
Strictly ignore page numbers, headers, footers, borders, bullets and any other elements that are not the main textual content of the document.\n
Maintain the original order of the text as it appears across all pages of the PDF. Do not rephrase or interpret the content; output the text verbatim. Pay attention to the sequence of paragraphs and sentences within each page.\n
Preserve the original line breaks within paragraphs and sentences as much as possible, unless they are clearly artifacts of page layout rather than sentence breaks. Maintain the original spacing between words.\n
The output should be a single, continuous string of text representing the content of the entire PDF, maintaining the original order of pages and the flow of text within each page.\n
Do not summarize, paraphrase, translate, or add any commentary to the extracted text. Your sole purpose is to provide the exact text as it exists in the PDF.\n"""

prompt2 ="""You are efficient and precise translator\n
Your receive English text in string. \n
Your task is to translate the text into Telugu.\n
Maintain the original order of the text as it appears.\n
Do not summarize or add any commentary to the translated text"""

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"
def analyze_pdf(doc, total_pages, min_image_area=10000):
    page_summaries = []
    text_only_pages = 0
    image_only_pages = 0
    mixed_pages = 0
    empty_pages = 0

    for page_num in range(total_pages):
        page = doc[page_num]
        text_blocks = page.get_text("dict")["blocks"]
        def has_real_text(block):
            if "lines" not in block:
                return False
            for line in block["lines"]:
                for span in line.get("spans", []):
                    if span["text"].strip():  # non-whitespace content
                        return True
            return False
        has_text = any(has_real_text(block) for block in text_blocks)


        # Use get_images to detect all image XObjects (embedded images)
        image_list = page.get_images(full=True)
        large_images = []

        for img in image_list:
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            area = pix.width * pix.height
            if area >= min_image_area:
                large_images.append(pix)
            pix = None  # cleanup

        has_image = len(large_images) > 0

        if has_text and not has_image:
            classification = "Text-only"
            text_only_pages += 1
        elif has_image and not has_text:
            classification = "Image-only"
            image_only_pages += 1
        elif has_text and has_image:
            classification = "Mixed (Text + Image)"
            mixed_pages += 1
        else:
            classification = "Empty or Unrecognized"
            empty_pages += 1

        page_summaries.append((page_num + 1, classification))

    doc.close()

    # Document-level classification
    if text_only_pages == total_pages:
        doc_classification = "Text-based"
    elif image_only_pages == total_pages:
        doc_classification = "Image-based"
    else:
        doc_classification = "Mixed content"

    return {
        "total_pages": total_pages,
        "page_summaries": page_summaries,
        "summary_counts": {
            "Text-only": text_only_pages,
            "Image-only": image_only_pages,
            "Mixed": mixed_pages,
            "Empty": empty_pages,
        },
        "document_classification": doc_classification
    }

# Example usage
def checkPdfFormat(filepath_str,output_filename):
    filepath = fitz.open(filepath_str)
    total_pages = len(filepath)
    result = analyze_pdf(filepath,total_pages)
    count=0
    print(f"Document Classification: {result['document_classification']}")
    print("Page Breakdown:")
    for page_num, classification in result["page_summaries"]:
        print(f"  Page {page_num}: {classification}")
        # if(classification=="Text-only"):
        #     count+=1
    extract_text(
            filepath_str, output_filename
        )
    # if(count==total_pages):
    #     extract_text(
    #         filepath_str, output_filename
    #     )
    # else:
    #     print("PDF is not text only")


def extract_text(filepath,output_filename):

    client = genai.Client(api_key=os.getenv("MY_API_KEY"))

    with open(filepath, "rb") as doc_file:
        doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")

    response = client.models.generate_content(
      model="gemini-1.5-flash",
      contents=[
          types.Part.from_bytes(
            data=doc_data,
            mime_type='application/pdf',
          ),
          prompt])

    with open(output_filename, "w", encoding="utf-8") as outfile:
        outfile.write(response.text)
    
    print(detect_language(response.text))
    detected_lang = detect_language(response.text)
    if(detected_lang=="en"):
        print("Language detected is english")
        response_translate = client.models.generate_content(
          model="gemini-1.5-flash",
          contents=[response.text,prompt2])
        with open("translate_output", "w", encoding="utf-8") as outfile:
            outfile.write(response_translate.text)
    else:
        print("Language detected is telugu")
def pdfinputType():
    pdfType = input("Enter 1.Telugu Text PDF 2.English Text PDF") 
    if pdfType=="1":
        print("1. Telugu")
        input_file = r"C:\LLMS_PROJECT\Input\puretel.pdf"
        output_file = "Telugu_extracted_text_GEMINI.txt"
        if not os.path.exists(input_file):
            print(f"PDF file '{input_file}' not found.")
        else:
            checkPdfFormat(input_file, output_file)
    else:
        print("2. english")
        input_file = r"C:\LLMS_PROJECT\Input\Photosynth.pdf"
        output_file = "English_extracted_text_GEMINI.txt"
        if not os.path.exists(input_file):
            print(f"PDF file '{input_file}' not found.")
        else:
            checkPdfFormat(input_file, output_file)

if __name__ == "__main__":
    # filepath_str = "Telugu.pdf"
    # filepath = fitz.open(filepath_str)
    # total_pages = len(filepath)
    # checkPdfFormat(filepath,total_pages,filepath_str)
    pdfinputType()
