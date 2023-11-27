import json
import os
import shutil
import requests

import gradio as gr
from huggingface_hub import Repository, InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN", None)
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-180B-chat"
BOT_NAME = "Falcon"

STOP_SEQUENCES = ["\nUser:", "<|endoftext|>", " User:", "###"]

EXAMPLES = [
    ["Hey Falcon! Any recommendations for my holidays in Abu Dhabi?"],
    ["What's the Everett interpretation of quantum mechanics?"],
    ["Give me a list of the top 10 dive sites you would recommend around the world."],
    ["Can you tell me more about deep-water soloing?"],
    ["Can you write a short tweet about the release of our latest AI model, Falcon LLM?"]
    ]

client = InferenceClient(
    API_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
)

def format_prompt(message, history, system_prompt):
  prompt = ""
  if system_prompt:
    prompt += f"System: {system_prompt}\n"
  for user_prompt, bot_response in history:
    prompt += f"User: {user_prompt}\n"
    prompt += f"Falcon: {bot_response}\n" # Response already contains "Falcon: "
  prompt += f"""User: {message}
Falcon:"""
  return prompt

seed = 42

def generate(
    prompt, history, system_prompt="", temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)
    global seed
    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_sequences=STOP_SEQUENCES,
        do_sample=True,
        seed=seed,
    )
    seed = seed + 1
    formatted_prompt = format_prompt(prompt, history, system_prompt)

    try:
        stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
        output = ""

        for response in stream:
            output += response.token.text
    
            for stop_str in STOP_SEQUENCES:
                if output.endswith(stop_str):
                    output = output[:-len(stop_str)]
                    output = output.rstrip()
                    yield output
            yield output
    except Exception as e:
        raise gr.Error(f"Error while generating: {e}")
    return output


additional_inputs=[
    gr.Textbox("", label="Optional system prompt"),
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=256,
        minimum=0,
        maximum=3000,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.01,
        maximum=0.99,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penalize repeated tokens",
    )
]


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=0.4):
            gr.Image("better_banner.jpeg", elem_id="banner-image", show_label=False)
        with gr.Column():
            gr.Markdown(
                """# Falcon-180B Demo

                **Chat with [Falcon-180B-Chat](https://huggingface.co/tiiuae/falcon-180b-chat), brainstorm ideas, discuss your holiday plans, and more!**
                
                ✨ This demo is powered by [Falcon-180B](https://huggingface.co/tiiuae/falcon-180B) and finetuned on a mixture of [Ultrachat](https://huggingface.co/datasets/stingning/ultrachat), [Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) and [Airoboros](https://huggingface.co/datasets/jondurbin/airoboros-2.1). [Falcon-180B](https://huggingface.co/tiiuae/falcon-180b) is a state-of-the-art large language model built by the [Technology Innovation Institute](https://www.tii.ae) in Abu Dhabi. It is trained on 3.5 trillion tokens (including [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)) and available under the [Falcon-180B TII License](https://huggingface.co/spaces/tiiuae/falcon-180b-license/blob/main/LICENSE.txt). It currently holds the 🥇 1st place on the [🤗 Open LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for a pretrained model. 
                
                🧪 This is only a **first experimental preview**: we intend to provide increasingly capable versions of Falcon in the future, based on improved datasets and RLHF/RLAIF.
                
                👀 **Learn more about Falcon LLM:** [falconllm.tii.ae](https://falconllm.tii.ae/)
                
                ➡️️ **Intended Use**: this demo is intended to showcase an early finetuning of [Falcon-180B](https://huggingface.co/tiiuae/falcon-180b), to illustrate the impact (and limitations) of finetuning on a dataset of conversations and instructions. We encourage the community to further build upon the base model, and to create even better instruct/chat versions!
                
                ⚠️ **Limitations**: the model can and will produce factually incorrect information, hallucinating facts and actions. As it has not undergone any advanced tuning/alignment, it can produce problematic outputs, especially if prompted to do so. Finally, this demo is limited to a session length of about 1,000 words.
                """
            )

    gr.ChatInterface(
        generate, 
        examples=EXAMPLES,
        additional_inputs=additional_inputs,
    ) 

demo.queue(concurrency_count=100, api_open=False).launch(show_api=False)
