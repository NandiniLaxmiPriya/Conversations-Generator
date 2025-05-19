import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from gtts import gTTS
load_dotenv()
client = genai.Client(api_key=os.getenv("MY_API_KEY"))

with open("translate_output.txt", "r", encoding="utf-8") as f:
    telugu_translated=f.read()
conv_prompt = f"""
Generate a realistic and engaging technical conversation in Telugu among three speakers based on the content provided in "{telugu_translated}".

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

file_path = r'C:\LLMS_PROJECT\conversations_output.txt'
with open(file_path, "r", encoding="utf-8") as f:
    text=f.read()

# Step 2: Convert text to speech
tts = gTTS(text=text, lang='te')

# Step 3: Save the audio file
audio_file = 'telugu_convo_audio.mp3'
tts.save(audio_file)

print(f"Audio content saved as {audio_file}")
