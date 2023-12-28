import openai
import os
import pandas as pd
import time
import json

service_endpoint = "" # eg. https://youraoairesource.opanai.azure.com/
service_key = ""


def read_prompt_data() -> dict:
    """
    Read in the prompt data from a json file
    """
    with open("test_prompts.json", "r") as json_file:
        test_prompts = json.load(json_file)
        json_file.close()
    return test_prompts

def call_openai_service(prompt) -> float:
    openai.api_type = "azure"
    openai.api_base = service_endpoint
    openai.api_version = "2023-07-01-preview"
    openai.api_key = service_key

    message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"{prompt}".format(prompt=prompt)}]
    print("Prompt:")
    print(prompt)
    print("Completion:")
    
    t1 = time.process_time()

    completion = openai.ChatCompletion.create(
    engine="gpt-35-turbo-16k",
    messages = message_text,
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
    )
    print(completion.choices[0]['message']['content'])
    
    t2 = time.process_time()
    
    print("====================================")
    return (t2-t1) * 1000

def main():
    test_prompts = read_prompt_data()
    results = []
    print("Warmup Azure OpenAI service...")
    call_openai_service("Hello, are you there?")
    for test_prompt in test_prompts["test_prompts"]:
        print("Testing category: {category}".format(category=test_prompt["category"]))
        for prompt in test_prompt["prompts"]:
            time_taken = call_openai_service(test_prompt["base_prompt"] + prompt["prompt"])
            results.append({"index": prompt["index"], "title": prompt["title"], "time_taken": time_taken})
        df = pd.DataFrame(results)
        #ret = df.groupby('index').agg({'time_taken': ['mean', 'min', 'max']})
    print(df)
    print("max = " + str(df.max(axis = 0)["time_taken"]) + "ms" + " min = " + str(df.min(axis = 0)["time_taken"]) + "ms" + " avg = " + str(df["time_taken"].mean()) + "ms")
    df.truncate(before=-1, after=-1, axis=0)

if __name__ == "__main__":
    main()