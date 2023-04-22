import json

import openai
import whisper

# load openai key from keys.json
try:
    with open("keys.json", 'r') as f:
        keys = json.load(f)
        openai.api_key = keys["openai"]
except Exception as e:
    print(e)
    exit(-1)


def audio_to_text(filename):
    model = whisper.load_model("large-v2")
    result = model.transcribe(filename)
    return (result["text"])


def send_gpt_request(transcribed_text, mode):
    print(f"Transcribed text: {transcribed_text}")
    if mode == "summarize":
        prompt = f"Please summarize the following text: {transcribed_text}"
    elif mode == "detect_tasks":
        prompt = f"Detect tasks and actions needed from the following text: {transcribed_text}"
    elif mode == "brainstorm":
        prompt = f"Generate brainstorming suggestions based on the following text: {transcribed_text}"
    else:
        raise ValueError("Invalid mode")
    
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    return completion.choices[0].message.content.strip()


def process_audio_file(filename, mode):
    transcribed_text = audio_to_text(filename)
    output_text = send_gpt_request(transcribed_text, mode)
    return output_text


if __name__ == "__main__":
    filename = "file1.wav"
    mode = "summarize"  # or "detect_tasks" or "brainstorm"

    result = process_audio_file(filename, mode)
    print(f"Result: {result}")
