import openai
from typing import List
import tiktoken
import time
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

total_tokens: int = 0
times: List[float] = []
token_length_list: List[int] = []

enc = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")

for i in range(30):
    start_time = time.time()
    msg = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "write a 200 word essay on math."}
        ]
    )
    end_time = time.time()
    times.append(float(end_time - start_time))

    return_length: int = len(enc.encode(msg.choices[0].message.content))
    assert return_length >= 200 # make sure it's at least 200 tokens. tokens != words
    token_length_list.append(return_length)
    total_tokens += return_length

tps_list: List[float] = []
for i in range(len(times)):
    tps_list.append(token_length_list[i] / times[i])

print("Average tokens per second: ", sum(tps_list) / len(tps_list))

print("min tokens per second: ", min(tps_list))

print("max tokens per second: ", max(tps_list))

