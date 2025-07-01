import os
from dotenv import load_dotenv
from google import genai
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from sacrebleu.metrics import BLEU as SacreBLEU
from nltk.tokenize import  word_tokenize
load_dotenv()

client = genai.Client(api_key=os.getenv("MY_API_KEY"))

with open("translate_output.txt", "r", encoding="utf-8") as f:
    telugu_translated=f.read()
prompt3 = f"""You are a precise translator. Translate the following Telugu text back to English.
        Preserve line breaks and sentence structure.\n\nTelugu Text: \"\"\"{telugu_translated}\"\"\""""
response_backtranslate = client.models.generate_content(
      model="gemini-1.5-flash",contents=[prompt3])
back_translated_english = response_backtranslate.text
with open("back_translated.txt", "w", encoding="utf-8") as outfile:
    outfile.write(back_translated_english)

with open("Gemini_extracted_text_output.txt", "r", encoding="utf-8") as f:
    reference_text=f.read()
reference = reference_text.strip()
candidate = back_translated_english.strip()

# BLEU using NLTK
reference_tokens = [reference.split()]
candidate_tokens = candidate.split()
smoothing = SmoothingFunction().method1
bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)

# BLEU using SacreBLEU
sacre_bleu = SacreBLEU()
sacre_bleu_score = sacre_bleu.corpus_score([candidate], [[reference]]).score

# METEOR
meteor_score = single_meteor_score(word_tokenize(reference), word_tokenize(candidate))


print("\n--- Evaluation Metrics ---")
print(f"NLTK BLEU Score: {bleu_score:.4f}")
print(f"SacreBLEU Score: {sacre_bleu_score:.2f}")
print(f"METEOR Score: {meteor_score:.4f}")