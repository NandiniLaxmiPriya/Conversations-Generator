import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import base64
import fitz
from langdetect import detect
load_dotenv()
prompt = """ You are an efficient and precise text-based PDF extractor.
You will receive the PDF, which may contain English or Telugu text. \n
Your task is to extract the complete textual content exactly as it appears in the PDF but You should not extract text from images, tables and figures. Preserving the original language of each segment.\n
You should extract the bullets and also extract the content for each bullet correctly. \n
Strictly ignore page numbers, headers, footers, borders and any other elements that are not the main textual content of the document.\n
Maintain the original order of the text as it appears across all pages of the PDF. Do not rephrase or interpret the content; output the text verbatim. Pay attention to the sequence of paragraphs and sentences within each page.\n
Preserve the original line breaks within paragraphs and sentences as much as possible, unless they are clearly artifacts of page layout rather than sentence breaks. Maintain the original spacing between words.\n
The output should be a single, continuous string of text representing the content of the entire PDF, maintaining the original order of pages and the flow of text within each page.\n
Do not summarize, paraphrase, translate, or add any commentary to the extracted text. Your sole purpose is to provide the exact text as it exists in the PDF.\n"""


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
    # count=0
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
    # print(response.text)
    print(detect_language(response.text))
    response_main=""
    detected_lang = detect_language(response.text)
    if(detected_lang=="en"):
        print("Language detected is english")
        prompt2 = f"""You are an efficient and precise translator.
                      Your task is to translate the following English text into Telugu but follow these rules:
1. Keep all technical terms in English without translating them.
2. Use simple and clear language in the translation so that a general audience can understand.
3. Do not change the structure of the original content too much — preserve the meaning.
4. If a sentence contains a technical phrase, only translate the rest of the sentence into the target language while keeping the term as-is
                      Preserve all original line breaks and paragraph structure exactly.
                      Do not summarize or add commentary.
                      Maintain sentence structure and order. Translate this English Text:\"\"\"{response.text}\"\"\""""
        response_translate = client.models.generate_content(
          model="gemini-1.5-flash",
          contents=[prompt2])
        telugu_translated = response_translate.text
        with open("translate_output.txt", "w", encoding="utf-8") as outfile:
            outfile.write(telugu_translated)
        response_main=telugu_translated
    else:
        print("Language detected is telugu")
        response_main=response.text
#     conv_prompt = f"""
# Generate a realistic and engaging technical conversation in telugu among three speakers from the content in the following "{response_main}".
# - Speaker 1 is an expert in the topic.
# - Speaker 2 has a moderate understanding.
# - Speaker 3 is a beginner and unfamiliar with the topic.
# The conversation should flow naturally, with:
# - Speaker 3 asking genuine beginner questions.
# - Speaker 2 attempting to explain in layman's terms and occasionally deferring to Speaker 1.
# - Speaker 1 providing in-depth insights and clarifying misconceptions.
# Do not specify speaker 1,2 or 3, instead specify speaker name. 
# The goal is for Speaker 3 to progressively understand the topic by the end of the conversation.
# Keep the conversation structured, informative, and accessible. Include technical explanations, analogies.
# If examples are given in the content, then use them else add examples where necessary.
# Do not give any extra information in English, before and after conversations.
# """
#     response_conversation = client.models.generate_content(
#           model="gemini-1.5-flash",
#           contents=[conv_prompt])
#     with open("conversations_output.txt", "w", encoding="utf-8") as outfile:
#             outfile.write(response_conversation.text)

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
        input_file = r"C:\LLMS_PROJECT\DS_test.pdf"
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
