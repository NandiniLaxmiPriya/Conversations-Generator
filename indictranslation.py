from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sacrebleu import corpus_bleu
import warnings

# Suppress warnings from transformers when loading with remote code
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- Model and Tokenizer Loading ---
model_name = "ai4bharat/indictrans2-en-indic-1B"
print(f"Loading model and tokenizer for: {model_name}...")
# The AutoTokenizer, when trust_remote_code=True, will load the custom tokenizer
# for IndicTrans2, which knows about the language tags like '<2en>', '<2te>', etc.
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded successfully on device: {device}")

# --- Translation Function ---
def translate(text, src_lang, tgt_lang):
    """
    Translates text from src_lang to tgt_lang using the IndicTrans2 model
    without relying on an explicit lang_map. It assumes the tokenizer will
    correctly handle the language codes passed directly in the input string.
    """
    # For IndicTrans2, the input format for translation is typically
    # "<src_lang_code> <tgt_lang_code> <text>" where the tokenizer
    # automatically converts these codes (e.g., 'eng_Latn') into
    # the internal special tokens (e.g., '<2en>').
    formatted_text = f"{src_lang} {tgt_lang} {text.strip()}"

    # Tokenize the formatted text. We explicitly do NOT pass src_lang and tgt_lang
    # as separate arguments to tokenizer() as its __call__ method doesn't recognize them.
    inputs = tokenizer(formatted_text, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate translation.
    with torch.no_grad(): # Disable gradient calculation for inference to save memory and speed up
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,        # Improves speed during generation
            min_length=1,          # Ensure at least one token is generated
            max_new_tokens=256,    # Controls the maximum length of the generated output
            num_beams=5,           # Use beam search for higher quality translations
            num_return_sequences=1, # Return only the best sequence
            early_stopping=True    # Stop when all beam hypotheses are complete
        )

    # Decode the generated tokens back to text, skipping special tokens.
    translations = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0] # [0] because num_return_sequences is 1

    return translations


# --- Text Chunking Function ---
def chunk_text(text, max_words=100):
    """
    Splits a long text into smaller chunks based on max_words.
    """
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# --- Main Translation and Evaluation Flow ---
if __name__ == "__main__":
    print("\n--- Starting Translation Process ---")
    # Read large English input
    try:
        with open("English_extracted_text_GEMINI.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print("Error: 'English_extracted_text_GEMINI.txt' not found. Please ensure the file exists.")
        exit()

    chunks = chunk_text(text)
    te_chunks, back_en_chunks = [], []

    # Translate chunks
    print(f"Translating {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        try:
            # English to Telugu translation using standard language codes
            te = translate(chunk, "eng_Latn", "tel_Telu")
            te_chunks.append(te)

            # Telugu to English back-translation
            back_en = translate(te, "tel_Telu", "eng_Latn")
            back_en_chunks.append(back_en)

            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"Processed {i + 1}/{len(chunks)} chunks.")

        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            print(f"Problematic chunk (first 100 chars): {chunk[:100]}...")
            te_chunks.append("")
            back_en_chunks.append("")

    print("\n--- Saving Translated Texts ---")
    # Save outputs
    with open("translated_te.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(te_chunks))
    print("Translated Telugu text saved to 'translated_te.txt'")

    with open("backtranslated_en.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(back_en_chunks))
    print("Back-translated English text saved to 'backtranslated_en.txt'")

    # --- Compute Evaluation Metrics ---
    print("\n--- Computing Evaluation Metrics ---")
    # Join all chunks back for corpus-level evaluation
    ref_text = " ".join(chunks)
    hyp_text = " ".join(back_en_chunks)

    # Tokenize for NLTK metrics
    ref_tokens = ref_text.split()
    hyp_tokens = hyp_text.split()

    # BLEU (NLTK sentence_bleu on concatenated text)
    smoothie = SmoothingFunction().method4
    bleu_score_nltk = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)

    # BLEU (SacreBLEU) - Recommended for corpus-level BLEU
    bleu_score_sacre = corpus_bleu([hyp_text], [[ref_text]]).score

    # METEOR (NLTK)
    meteor = meteor_score([ref_tokens], hyp_tokens)

    print(f"NLTK BLEU (on concatenated text): {bleu_score_nltk:.4f}")
    print(f"SacreBLEU (Corpus BLEU): {bleu_score_sacre:.2f}")
    print(f"METEOR (on concatenated text): {meteor:.4f}")
    print("\n--- Translation and Evaluation Complete ---")