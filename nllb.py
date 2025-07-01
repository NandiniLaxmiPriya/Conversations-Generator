# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import sacrebleu
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.translate.meteor_score import single_meteor_score

# nltk.download('punkt')

# # Load model and tokenizer
# model_name = "facebook/nllb-200-distilled-600M"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# # Language codes
# src_lang = "eng_Latn"
# mid_lang = "tel_Telu"

# # File paths
# input_file = r"C:\LLMS_PROJECT\English_extracted_text_GEMINI.txt"
# output_telugu_file = "nllb_output_telugu.txt"
# output_back_file = "nllb_back_translated_english.txt"

# # Sentence splitting and token-safe batching
# def batch_sentences(sentences, tokenizer, max_tokens=512):
#     batches, current_batch, current_len = [], [], 0
#     for sent in sentences:
#         tokens = tokenizer.tokenize(sent)
#         if current_len + len(tokens) > max_tokens:
#             if current_batch:
#                 batches.append(" ".join(current_batch))
#                 current_batch, current_len = [], 0
#         current_batch.append(sent)
#         current_len += len(tokens)
#     if current_batch:
#         batches.append(" ".join(current_batch))
#     return batches

# # Read paragraphs
# def read_paragraphs(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         paragraph = []
#         for line in f:
#             if line.strip() == "":
#                 if paragraph:
#                     yield " ".join(paragraph).strip()
#                     paragraph = []
#             else:
#                 paragraph.append(line.strip())
#         if paragraph:
#             yield " ".join(paragraph).strip()

# # BLEU/METEOR containers
# original_english = []
# back_translated_english = []

# with open(output_telugu_file, "w", encoding="utf-8") as tel_out, \
#      open(output_back_file, "w", encoding="utf-8") as back_out:

#     for i, paragraph in enumerate(read_paragraphs(input_file), 1):
#         if not paragraph.strip():
#             continue

#         # === Sentence Tokenization + Safe Batching ===
#         sentences = sent_tokenize(paragraph)
#         batches = batch_sentences(sentences, tokenizer)

#         # --- English → Telugu ---
#         tokenizer.src_lang = src_lang
#         telugu_batches = []
#         for batch in batches:
#             encoded = tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(model.device)
#             outputs = model.generate(
#                 **encoded,
#                 forced_bos_token_id=tokenizer.convert_tokens_to_ids(mid_lang),
#                 max_length=512
#             )
#             telugu_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#             telugu_batches.append(telugu_text)
#         telugu_paragraph = " ".join(telugu_batches)
#         tel_out.write(telugu_paragraph + "\n\n")

#         # --- Telugu → English ---
#         tokenizer.src_lang = mid_lang
#         back_batches = []
#         for tel_batch in sent_tokenize(telugu_paragraph):  # back-split in case Telugu gets long
#             encoded_back = tokenizer(tel_batch, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(model.device)
#             outputs_back = model.generate(
#                 **encoded_back,
#                 forced_bos_token_id=tokenizer.convert_tokens_to_ids(src_lang),
#                 max_length=512
#             )
#             back_text = tokenizer.batch_decode(outputs_back, skip_special_tokens=True)[0]
#             back_batches.append(back_text)
#         back_paragraph = " ".join(back_batches)
#         back_out.write(back_paragraph + "\n\n")

#         # Collect for evaluation
#         original_english.append(paragraph)
#         back_translated_english.append(back_paragraph)

#         print(f"✔️ Paragraph {i} translated & back-translated.")

# # Evaluation
# bleu = sacrebleu.corpus_bleu(back_translated_english, [original_english])
# print(f"\n🔵 BLEU Score: {bleu.score:.2f}")

# meteor_scores = [
#     single_meteor_score(word_tokenize(orig), word_tokenize(back))
#     for orig, back in zip(original_english, back_translated_english)
# ]
# avg_meteor = sum(meteor_scores) / len(meteor_scores)
# print(f"🟢 METEOR Score: {avg_meteor:.4f}")

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# import re
# from nltk.tokenize import sent_tokenize

# # Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
# model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Language codes
# en_lang = "eng_Latn"
# te_lang = "tel_Telu"

# # Telugu sentence tokenizer
# def telugu_sent_tokenize(text):
#     return [s.strip() for s in re.split(r'(?<=[.?!।])\s+', text) if s.strip()]

# # Remove repeated Telugu sentences
# def remove_repetitions(text):
#     lines = text.split('.')
#     seen = set()
#     result = []
#     for line in lines:
#         line = line.strip()
#         if line and line not in seen:
#             result.append(line)
#             seen.add(line)
#     return '. '.join(result)

# # Translate a sentence or block
# def translate_text(text, src_lang, tgt_lang):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
#     outputs = model.generate(
#         **inputs,
#         forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
#         max_length=512,
#         num_beams=4,
#         early_stopping=True
#     )
#     return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# # === Main pipeline ===
# with open("English_extracted_text_GEMINI.txt", "r", encoding="utf-8") as f:
#     original_text = f.read().strip()

# # English to Telugu
# telugu_text = translate_text(original_text, en_lang, te_lang)
# telugu_text_cleaned = remove_repetitions(telugu_text)

# # Telugu to English (back-translation)
# back_translated_text = translate_text(telugu_text_cleaned, te_lang, en_lang)

# # Write Telugu output
# with open("nllb_output_telugu.txt", "w", encoding="utf-8") as f:
#     f.write(telugu_text_cleaned)

# # Write back-translated English
# with open("nllb_back_translated_english.txt", "w", encoding="utf-8") as f:
#     f.write(back_translated_text)

# # === Evaluation on full text ===
# # BLEU
# smooth_fn = SmoothingFunction().method4
# bleu = sentence_bleu([original_text.split()], back_translated_text.split(), smoothing_function=smooth_fn)

# # METEOR
# meteor = meteor_score([original_text.split()], back_translated_text.split())

# # Print scores
# print(f"BLEU Score: {bleu:.4f}")
# print(f"METEOR Score: {meteor:.4f}")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import sent_tokenize
from sacrebleu.metrics import BLEU as SacreBLEU

# Download punkt tokenizer if not already
# nltk.download('punkt')

# Load NLLB model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_text(text, src_lang_code, tgt_lang_code):
    tokenizer.src_lang = src_lang_code
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get forced BOS token id for target language
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
    if forced_bos_token_id is None:
        # fallback: encode the language code as token(s) and take first token id
        forced_bos_token_id = tokenizer.encode(tgt_lang_code, add_special_tokens=False)[0]

    generated_tokens = model.generate(
        **encoded, 
        forced_bos_token_id=forced_bos_token_id
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


# Read input from file
with open("Gemini_extracted_text_output.txt", "r", encoding="utf-8") as f:
    original_text = f.read()

# Sentence tokenization
sentences = sent_tokenize(original_text)

telugu_sentences = []
back_translated_sentences = []

for sentence in sentences:
    if len(sentence.strip()) < 3:
        continue  # Skip empty or very short lines
    telugu = translate_text(sentence, "eng_Latn", "tel_Telu")
    back = translate_text(telugu, "tel_Telu", "eng_Latn")
    telugu_sentences.append(telugu)
    back_translated_sentences.append(back)

# Write outputs to files
with open("nllb_output_telugu.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(telugu_sentences))

with open("nllb_back_translated_english.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(back_translated_sentences))

# Combine for evaluation
combined_original = " ".join(sentences)
combined_back_translated = " ".join(back_translated_sentences)

# BLEU
smoothie = SmoothingFunction().method4
bleu = sentence_bleu(
    [combined_original.split()],  # reference
    combined_back_translated.split(),  # hypothesis
    smoothing_function=smoothie
)

# BLEU using SacreBLEU
sacre_bleu = SacreBLEU()
sacre_bleu_score = sacre_bleu.corpus_score([combined_original], [[combined_back_translated]]).score

combined_original = " ".join(sentences)
combined_back_translated = " ".join(back_translated_sentences)

meteor = meteor_score([combined_original.split()], combined_back_translated.split())

# Print scores
print(f"\n NLTK BLEU score: {bleu:.4f}")
print(f"METEOR score: {meteor:.4f}")
print(f"SacreBLEU Score: {sacre_bleu_score:.2f}")
