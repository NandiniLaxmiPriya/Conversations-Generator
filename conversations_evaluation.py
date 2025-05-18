import os
from dotenv import load_dotenv
from google import genai
from gtts import gTTS
load_dotenv()
client = genai.Client(api_key=os.getenv("MY_API_KEY"))

with open("translate_output.txt", "r", encoding="utf-8") as f:
    telugu_translated=f.read()
conv_prompt = f"""
Generate a realistic and engaging technical conversation in telugu among three speakers from the content in the following "{telugu_translated}".
- Speaker 1 is an expert in the topic.
- Speaker 2 has a moderate understanding.
- Speaker 3 is a beginner and unfamiliar with the topic.
The conversation should flow naturally, with:
- Speaker 3 asking genuine beginner questions.
- Speaker 2 attempting to explain in layman's terms and occasionally deferring to Speaker 1.
- Speaker 1 providing in-depth insights and clarifying misconceptions.
Do not specify speaker 1,2 or 3, instead specify speaker name. 
The goal is for Speaker 3 to progressively understand the topic by the end of the conversation.
Keep the conversation structured, informative, and accessible. Include technical explanations, analogies.
If examples are given in the content, then use them else add examples where necessary.
Do not give any extra information in English, before and after conversations.
Maintain a Natural Tone: The conversation should sound like a real discussion, not a lecture.
Do not add **speaker name**, just keep speaker name
"Output format: Provide the character names,description of characters, and then the conversation.
"""
response_conversation = client.models.generate_content(
          model="gemini-1.5-flash",
          contents=[conv_prompt])
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
