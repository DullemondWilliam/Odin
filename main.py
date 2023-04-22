import json

import openai
import whisper
import tiktoken

# load openai key from keys.json
try:
    with open("keys.json", 'r') as f:
        keys = json.load(f)
        openai.api_key = keys["openai"]
except Exception as e:
    print(e)
    exit(-1)

MODEL = "gpt-3.5-turbo"
MODEL_MAX_TOKENS = 4096

def audio_to_text(filename):
    model = whisper.load_model("large-v2")
    result = model.transcribe(filename)
    
    text_chunks = chunk_text(result)
    
    return (result["text"])


def chunk_text(text_result, prompt_buffer=50, response_buffer=500):
    '''
    Takes result of whisper transcription and returns a list of text chunks.
     
    The size of these chunks are based on the MODEL and MODEL_MAX_TOKENS
    such that prompt_buffer + number of tokens of chunk + response_buffer < MODEL_MAX_TOKENS
    '''

    chunks = []
    _chunk_text = ''
    chunk_tokens = prompt_buffer + response_buffer
    encoder = tiktoken.encoding_for_model(MODEL)
    for segment in text_result.get('segments'):
        segment_text = segment.get('text')
        segment_tokens = len(encoder.encode(segment_text, disallowed_special=()))

        if chunk_tokens + segment_tokens > MODEL_MAX_TOKENS:
            chunks.append(_chunk_text)
            _chunk_text = ''
            chunk_tokens = prompt_buffer + response_buffer
        
        chunk_tokens += segment_tokens
        _chunk_text += segment_text

    if len(_chunk_text) > 0:
        chunks.append(_chunk_text)

    return chunks


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
    
    completion = openai.ChatCompletion.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
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
