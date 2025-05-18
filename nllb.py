import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.meteor_score import single_meteor_score

nltk.download('punkt')

# Load model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Language codes
src_lang = "eng_Latn"
mid_lang = "tel_Telu"

# File paths
input_file = r"C:\LLMS_PROJECT\English_extracted_text_GEMINI.txt"
output_telugu_file = "nllb_output_telugu.txt"
output_back_file = "nllb_back_translated_english.txt"

# Sentence splitting and token-safe batching
def batch_sentences(sentences, tokenizer, max_tokens=512):
    batches, current_batch, current_len = [], [], 0
    for sent in sentences:
        tokens = tokenizer.tokenize(sent)
        if current_len + len(tokens) > max_tokens:
            if current_batch:
                batches.append(" ".join(current_batch))
                current_batch, current_len = [], 0
        current_batch.append(sent)
        current_len += len(tokens)
    if current_batch:
        batches.append(" ".join(current_batch))
    return batches

# Read paragraphs
def read_paragraphs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        paragraph = []
        for line in f:
            if line.strip() == "":
                if paragraph:
                    yield " ".join(paragraph).strip()
                    paragraph = []
            else:
                paragraph.append(line.strip())
        if paragraph:
            yield " ".join(paragraph).strip()

# BLEU/METEOR containers
original_english = []
back_translated_english = []

with open(output_telugu_file, "w", encoding="utf-8") as tel_out, \
     open(output_back_file, "w", encoding="utf-8") as back_out:

    for i, paragraph in enumerate(read_paragraphs(input_file), 1):
        if not paragraph.strip():
            continue

        # === Sentence Tokenization + Safe Batching ===
        sentences = sent_tokenize(paragraph)
        batches = batch_sentences(sentences, tokenizer)

        # --- English → Telugu ---
        tokenizer.src_lang = src_lang
        telugu_batches = []
        for batch in batches:
            encoded = tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(model.device)
            outputs = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(mid_lang),
                max_length=512
            )
            telugu_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            telugu_batches.append(telugu_text)
        telugu_paragraph = " ".join(telugu_batches)
        tel_out.write(telugu_paragraph + "\n\n")

        # --- Telugu → English ---
        tokenizer.src_lang = mid_lang
        back_batches = []
        for tel_batch in sent_tokenize(telugu_paragraph):  # back-split in case Telugu gets long
            encoded_back = tokenizer(tel_batch, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(model.device)
            outputs_back = model.generate(
                **encoded_back,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(src_lang),
                max_length=512
            )
            back_text = tokenizer.batch_decode(outputs_back, skip_special_tokens=True)[0]
            back_batches.append(back_text)
        back_paragraph = " ".join(back_batches)
        back_out.write(back_paragraph + "\n\n")

        # Collect for evaluation
        original_english.append(paragraph)
        back_translated_english.append(back_paragraph)

        print(f"✔️ Paragraph {i} translated & back-translated.")

# Evaluation
bleu = sacrebleu.corpus_bleu(back_translated_english, [original_english])
print(f"\n🔵 BLEU Score: {bleu.score:.2f}")

meteor_scores = [
    single_meteor_score(word_tokenize(orig), word_tokenize(back))
    for orig, back in zip(original_english, back_translated_english)
]
avg_meteor = sum(meteor_scores) / len(meteor_scores)
print(f"🟢 METEOR Score: {avg_meteor:.4f}")
