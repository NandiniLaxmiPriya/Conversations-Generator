import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import base64
import fitz
from langdetect import detect
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
client = genai.Client(api_key=os.getenv("MY_API_KEY"))
app = FastAPI()

# Allow frontend (Next.js) to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # adjust as needed
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

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
def checkPdfFormat(pdf_bytes: bytes,output_filename: str):
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_doc)
    result = analyze_pdf(pdf_doc, total_pages)
    # count=0
    print(f"Document Classification: {result['document_classification']}")
    print("Page Breakdown:")
    for page_num, classification in result["page_summaries"]:
        print(f"  Page {page_num}: {classification}")
    final_result = extract_text(
            pdf_bytes, output_filename
        )
    return final_result


def extract_text(pdf_bytes: bytes, output_filename: str):

    # with open(filepath, "rb") as doc_file:
    #     doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")

    response = client.models.generate_content(
      model="gemini-1.5-flash",
      contents=[
          types.Part.from_bytes(
            data=pdf_bytes,
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
    result_convo = conversations_generator(response_main)
    return result_convo


def conversations_generator(response_main):
    conv_prompt = f"""
Generate a realistic and engaging technical conversation in Telugu among three speakers based on the content provided in "{response_main}".

Goals:
- The conversation should fully explain the entire content without skipping any paragraph or sub-point.
- The discussion should flow naturally, as a single continuous conversation — not as isolated exchanges for each paragraph.
- All key points, paragraphs, and sub-topics must be covered progressively throughout the dialogue.

Character Roles:
- Speaker 1 is an expert in the topic.
- Speaker 2 has a moderate understanding and explains ideas in layman’s terms.
- Speaker 3 is a beginner and asks simple, genuine questions to learn about the topic.

Conversation Style:
- Speaker 3 initiates or responds with beginner-level questions or confusion.
- Speaker 2 gives simplified explanations and sometimes refers to Speaker 1 for clarity.
- Speaker 1 provides deeper, accurate insights, corrects any misunderstandings, and explains technical points clearly.
- Include relevant examples: use those present in the text; if not, add your own for better understanding.
- Use analogies and real-world references where helpful.

Instructions:
- The conversation must be structured, informative, and engaging.
- Ensure the flow of dialogue naturally introduces and explains the content in the order it appears.
- Do not skip any topics or subpoints.
- Do not present the text as a summary or narration — keep it as a natural back-and-forth discussion.
- Do not include any English explanation before or after the conversation.
- Do not prefix speakers with labels like **Speaker 1**, just use their character names.

Output Format:
1. Start with character names and a one-line description for each.
2. Then write the conversation, ensuring full coverage of the content in a natural tone.

"""
    response_conversation = client.models.generate_content(
          model="gemini-1.5-flash",
          contents=[conv_prompt],
          config=types.GenerateContentConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=1
    ))
    with open("conversations_output.txt", "w", encoding="utf-8") as outfile:
        outfile.write(response_conversation.text)
    return {"conversation": response_conversation.text}


@app.post("/upload")
async def pdfinputType(file: UploadFile = File(...)):
    # pdfType = input("Enter 1.Telugu Text PDF 2.English Text PDF") 
    # if pdfType=="1":
    #     print("1. Telugu")
    #     input_file = r"C:\LLMS_PROJECT\Input\puretel.pdf"
    #     output_file = "Telugu_extracted_text_GEMINI.txt"
    #     if not os.path.exists(input_file):
    #         print(f"PDF file '{input_file}' not found.")
    #     else:
    #         checkPdfFormat(input_file, output_file)
    # else:
    #     print("2. english")
    #     input_file = "DS_test.pdf"
    #     output_file = "English_extracted_text_GEMINI.txt"
    #     if not os.path.exists(input_file):
    #         print(f"PDF file '{input_file}' not found.")
    #     else:
    #         checkPdfFormat(input_file, output_file)
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # ✅ Read file contents in memory (no disk I/O)
    file_bytes = await file.read()

    # ✅ Call processing function
    result= checkPdfFormat(file_bytes,output_filename="Gemini_extracted_text_output.txt")
    return {"detail": f"{file.filename} uploaded and processed successfully", "data":result}

if __name__ == "__main__":
    # filepath_str = "Telugu.pdf"
    # filepath = fitz.open(filepath_str)
    # total_pages = len(filepath)
    # checkPdfFormat(filepath,total_pages,filepath_str)
    print("Started!")
