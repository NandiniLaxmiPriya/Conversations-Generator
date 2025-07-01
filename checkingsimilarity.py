from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_semantic_similarity(text1, text2, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    try:
        model = SentenceTransformer(model_name)
        embeddings1 = model.encode(text1,convert_to_tensor=False)
        embeddings2 = model.encode(text2,convert_to_tensor=False)
        similarity_score = cosine_similarity([embeddings1], [embeddings2])[0][0]
        return similarity_score
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def read_text_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
        return None

if __name__ == "__main__":
    # Specify the file paths for the text extracted by PyMuPDF and Gemini API
    # pdfType = input("Enter 1.Telugu Text PDF 2.English Text PDF") 
    # if pdfType=="1":
    #     print("1. Telugu")
    #     input1_text_file = "English_extracted_text_GEMINI.txt"
    #     input2_text_file="Output.txt"
    # else:
        # print("2. english")
    input1_text_file = 'translate_output.txt'
    input2_text_file = 'conversations_output.txt'
    
        

    # Read the extracted text from the files
    input2_text = read_text_from_file(input2_text_file)
    input1_text = read_text_from_file(input1_text_file)

    if input2_text is not None and input1_text is not None:
        # Calculate the semantic similarity using SBERT
        similarity_score = calculate_semantic_similarity(input2_text, input1_text)

        if similarity_score is not None:
            print(f"Semantic Similarity between text1 and text2 (using paraphrase-multilingual-MiniLM-L12-v2): {similarity_score:.4f}")
        else:
            print("Could not calculate semantic similarity.")
    else:
        print("Could not read text from one or both of the files.")