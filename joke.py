import os
import json
import time

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# === CONFIGURATION ===
INPUT_FILE = 'common_words.txt'
OUTPUT_FILE = 'meanings.txt'
BATCH_SIZE = 100
DELAY = 2  # seconds between batches to respect rate limits

# === Azure OpenAI Setup ===
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-mini"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

# === Load words ===
with open(INPUT_FILE, 'r', encoding='utf-8') as file:
    words = [line.strip() for line in file if line.strip()]

# === Batch Process ===
for i in range(0, len(words), BATCH_SIZE):
    batch = words[i:i + BATCH_SIZE]

    # ==== Create prompt ====
    batch_text = ', '.join(batch)
    user_prompt = f"""
Given the following list of adjectives:

{batch_text}

Return a JSON array where each object has the following structure:
{{
    "id": "1",
    "Noun": "eat",
    "Meaning": "",
    "Question": "Samuel always demonstrates the capability to handle complex tasks. What adjective best describes him?"
}}

Return only the JSON array.
    """.strip()

    # ==== Make API Call ====
    try:
        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant that provides adjective meanings and usage."),
                UserMessage(content=user_prompt)
            ],
            temperature=1,
            top_p=1,
            model=model
        )

        result_text = response.choices[0].message.content.strip()

        # Try to load JSON from the response
        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError:
            print(f"❌ Could not parse JSON for batch {i//BATCH_SIZE + 1}. Skipping.")
            continue

        # Write results to file
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
            for item in result_json:
                json_line = json.dumps(item, ensure_ascii=False)
                outfile.write(json_line + '\n')

        print(f"✅ Batch {i//BATCH_SIZE + 1} done: {len(batch)} words")
        time.sleep(DELAY)

    except Exception as e:
        print(f"❌ Error in batch {i//BATCH_SIZE + 1}: {e}")
        break

