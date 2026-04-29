import argparse
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
import re
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from itertools import islice

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def process_text(text: str) -> str:
    text = re.sub(r'<span\s+class="quote">.*?</span>', '', text, flags=re.DOTALL)
    # remove HTML tags
    #text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'<a[^>]*class="quotelink"[^>]*>.*?</a>', '', text)

    # remove special characters
    text = re.sub(r'&#039;', "'", text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)

    return text.strip()

def extract_post_id(text: str) -> str:
    match = re.search(r'class="quotelink"[^>]*>.*?&gt;&gt;(\d+)<', text)
    if match:
        return match.group(1)
    return str(hash(text))[:10]  # if no reference ID, use the hash value of the text as ID

def text_to_persona(text: str) -> str:
    system_prompt = """Analyze who would write this text. 
    IMPORTANT: Start your response with the persona description directly.
    DO NOT start with phrases like:
    - "The text is likely to be written by"
    - "This text appears to be written by"
    - "The author is likely"
    Just describe the persona directly."""
    
    user_prompt = f"""Based on this text, describe the persona who would write it:
    {text}
    Remember: Start your description directly without any introductory phrases!"""
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def process_single_post(post):
    if "perspectives" not in post or "com" not in post:
        return None
        
    if post["perspectives"].get("TOXICITY", 0) > 0.7:
        cleaned_text = process_text(post["com"])
        if len(cleaned_text) > 500:
            post_id = extract_post_id(post["com"])
            persona = text_to_persona(cleaned_text)
            if persona:
                return {
                    "post_id": post_id,
                    "cleaned_post": cleaned_text,
                    "perspectives": post.get("perspectives", {}),
                    "persona": persona
                }
    return None

def process_chunk(chunk_data):
    results = []
    try:
        data = json.loads(chunk_data)
        if "posts" not in data:
            return results
            
        for post in data["posts"]:
            result = process_single_post(post)
            if result:
                results.append(result)
    except json.JSONDecodeError as e:
        print(f"\nError decoding JSON chunk: {e}")
    except Exception as e:
        print(f"\nError processing chunk: {e}")
    return results

def process_ndjson(file_path: str, output_path: str, sample_size: int = None, num_workers: int = None):
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = list(islice(f, 0, sample_size) if sample_size else f)
    
    total_lines = len(lines)
    total_processed = 0
    
    # open file in 'w' mode
    with open(output_path, 'w', encoding='utf-8') as out, \
         ProcessPoolExecutor(max_workers=num_workers) as executor:
        for results in tqdm(executor.map(process_chunk, lines), 
                          total=total_lines,
                          desc="Processing posts"):
            # write each processed result in real-time
            for result in results:
                out.write(json.dumps(result, ensure_ascii=False) + '\n')
                out.flush() 
                total_processed += 1
    
    print(f"\nProcessed {total_processed} posts from {total_lines} chunks")

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=config["input_file"])
    parser.add_argument("--output_path", type=str, default=config["output_path"])
    parser.add_argument("--sample_size", type=int, default=config["sample_size"])
    parser.add_argument("--num_workers", type=int, default=None,
                      help="Number of worker processes to use. Defaults to CPU count - 1")
    args = parser.parse_args()
    
    process_ndjson(args.input_file, args.output_path, args.sample_size, args.num_workers)

if __name__ == "__main__":
    main()