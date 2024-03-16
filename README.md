# Takehome-PersonaAI
TakeHome for Persona AI - Implementation of Prompt Caching in an adaptation of python llama

## Task Definition 

Your task is to build an input caching layer for a 7B parameter LLM (we recommend Mistral or Llama).

### Context
LLMs have some cost to process the input tokens even before any output tokens are emitted - often referred to as the “prefill” phase. This is before the decode phase runs. Even though they are less expensive to process than output tokens, you typically have far more of them. With a long prompt, this can be fairly slow. This matter is exacerbated in typical ‘assistant’ style conversations as the conversation grows longer and longer over time. So a typical scaled production approach is to cache some of this work (model state) done during the prefill phase so you can just update from there the next human line comes in.

We want you to implement a version of this using PyTorch (feel free to use HuggingFace models). Effectively, you will want the ability to ‘save’ the (only necessary) state of the LLM and return a handle to reference it later, and another function that will perform full inferencing given the handle and additional text to add to the prompt. Think carefully about the design so that it works with these use cases:

Typical chat/assistant style where new lines are added to the prompt (so you will want to save after every inference)
Preloading ‘instruct’ style prompts for a task containing the overall task instructions but not the input, so you can just append the input later and run inferencing.
Your system should support multiple saved ‘states’, although you can certainly cap it to a small finite number (e.g. 8). Only save what is necessary (rather than the entire model) to perform inferencing from that state.

## Setup Details

- Run ```pip install -r requirements.txt``` (To install all requirements)
- Run ```bash download.sh``` to run the LLama2 download script and download LLama 7B. (Can download others and change the path in inference.py). This requires a presigned url with access to LLama-7B
- Thats it! you can now run the two commands for benchmarking or edit and run the inference.py file using the functions provided

  ```python inference.py --task system``` (This is for the instruct style prompt cache benchmarking)

  ```python inference.py --task chat``` (This is for the chat style prompt cache benchmarking)

## Implementation Details 

- 3 Major functions that are part of the model - `cache_prompt` and `text_completion` (used for instruct flow) and `chat_completion` (used for chat flow) to generate text using the caching system
- `cache_id` parameter passed in to each function to which part of the specific cache that needs to point to (ability to store multiple copies or different states at the same time)
- Using KV (Key Value) Caching mechanism and augmenting it to checkpoint various states of the attention across multiple layers.

## General Benchmarking Results
### Instruct Style Benchmarking - 
- Cache setup time: 15.76861310005188s
- Completion time with cache (Added Input Prompt 1): 0.7743737697601318s
- Completion time with cache (Added Input Prompt 2): 0.9078752994537354s

- Completion time without cache (Prompt 1): 15.868504047393799s
- Completion time without cache (Prompt 2): 16.007229566574097s

Time saved by caching: 14.43s

### Chat Style Benchmarking
- Initial Chat time (Chat1): 2.77986741065979s
- Completion time with cache (Chat 2): 0.3882172107696533s
- Completion time with cache (Chat 3): 0.38900065422058105s

- Completion time for without cache for chat 1 and 2: 2.3885085582733154s
- Completion time without cache for chat 1,2,3: 2.741135358810425s

Time saved by caching: 4.344s

## Future work

There is specific feature where after generation a state of the chat can be restored to the previous_state(s) (This was added as an extra functionality for things like regenerate in ChatGPT) but was not implemented due to the overhead (dictionary of dictionaries) and extra storage required.

Implementation of more modular caching for even lower latencies used in this paper - https://arxiv.org/abs/2311.04934 (Prompt Cache paper)

## Credits

Adapted from Meta LLama repository and specifically https://github.com/hkproj/pytorch-llama (A python implememtation of llama2 in pytorch)
