from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
import argparse

from tqdm import tqdm

from model import ModelArgs, Transformer

class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
        self.cached_id_lengths = {}
        self.cached_chat_lengths = {}

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location=device)
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        
        model = Transformer(model_args).to(device)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        return LLaMA(model, tokenizer, model_args)

    def text_completion(self, prompts: list[str], temperature: float = 0, top_p: float = 0.9, max_gen_len: Optional[int] = None, cache_id=None):
        if cache_id and self.cached_id_lengths.get(cache_id) is not None:
          token_cache_length = self.cached_id_lengths[cache_id]
          self.model.set_from_cache(cache_id)
        else:
          token_cache_length = 0
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len+token_cache_length)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
        cur_iterator = tqdm(range(token_cache_length+1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1-token_cache_length:cur_pos-token_cache_length], cur_pos, cache_id)
            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos-token_cache_length], tokens[:, cur_pos-token_cache_length], next_token)
            tokens[:, cur_pos-token_cache_length] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos-token_cache_length]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            final_prompt_tokens = [x for x in current_prompt_tokens if x != -1]
            if self.tokenizer.eos_id in final_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                final_prompt_tokens = final_prompt_tokens[:eos_idx]
            out_tokens.append(final_prompt_tokens)
            out_text.append(self.tokenizer.decode(final_prompt_tokens))
        return (out_tokens, out_text)


    def chat_completion(self, prompts: list[str], temperature: float = 0, top_p: float = 0.9, max_gen_len: Optional[int] = None, cache_id=None):
        if cache_id and self.cached_id_lengths.get(cache_id) is not None:
          token_cache_length = self.cached_id_lengths[cache_id]
          self.model.set_from_cache(cache_id)
        else:
          token_cache_length = 0
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len + token_cache_length)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
        cur_iterator = tqdm(range(token_cache_length+1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1-token_cache_length:cur_pos-token_cache_length], cur_pos, cache_id, to_cache=True)
            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos-token_cache_length], tokens[:, cur_pos-token_cache_length], next_token)
            tokens[:, cur_pos-token_cache_length] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos-token_cache_length]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            final_prompt_tokens = [x for x in current_prompt_tokens if x != -1]
            if self.tokenizer.eos_id in final_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                final_prompt_tokens = final_prompt_tokens[:eos_idx]
            out_tokens.append(final_prompt_tokens)
            out_text.append(self.tokenizer.decode(final_prompt_tokens))
        if cache_id:
          self.cached_id_lengths[cache_id] = len(out_tokens[0]) + token_cache_length        
        return (out_tokens, out_text)


    def cache_prompt(self, prompts: list[str], temperature: float = 0, top_p: float = 0.9, max_gen_len: Optional[int] = 0, cache_id=None):
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos, cache_id, to_cache=True)
            # Greedily select the token with the max probability
            next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            final_prompt_tokens = [x for x in current_prompt_tokens if x != -1]
            if self.tokenizer.eos_id in final_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                final_prompt_tokens = final_prompt_tokens[:eos_idx]
            out_tokens.append(final_prompt_tokens)
        if cache_id:
          self.cached_id_lengths[cache_id] = len(out_tokens[0])        
        return None
    
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p 
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0 
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token



if __name__ == '__main__':
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description="Run inference tasks.")
    parser.add_argument('--task', type=str, choices=['system', 'chat'], required=True, help='The task to run: "emotion" for emotion classification or "chat" for chat completion.')

    args = parser.parse_args()

    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=1,
        device=device
    )
    if args.task == 'chat':
      chat_prompt_init = """Let's play a game with numbers. For each number add the previous two generated number to this and return it
      Examples:
      1
      
      1
      
      2
      
      3
      
      5
      
      8
      
      """
      chat_next_line ="""
      21
      
      """

      chat_last_line ="""
      55
      
      """
      
     # Cache prompt and measure time
      start_time = time.time()
      model.chat_completion([chat_prompt_init], cache_id=1, max_gen_len=3)
      cache_time = time.time() - start_time

      # Measure time for completion with caching
      start_time = time.time()
      _, out_texts_cache_1 = model.chat_completion([chat_next_line], max_gen_len=3, cache_id=1)
      time_cache_1 = time.time() - start_time

      start_time = time.time()
      _, out_texts_cache_2 = model.chat_completion([chat_last_line], max_gen_len=3, cache_id=1)
      time_cache_2 = time.time() - start_time


      # Measure time for completion without caching
      start_time = time.time()
      _, out_texts_no_cache_1 = model.text_completion([chat_prompt_init +'\n13'+chat_next_line], max_gen_len=3)
      time_no_cache_1 = time.time() - start_time

      start_time = time.time()
      _, out_texts_no_cache_2 = model.text_completion([chat_prompt_init +'\n13'+chat_next_line +'\n34'+chat_last_line], max_gen_len=3)
      time_no_cache_2 = time.time() - start_time

      # Print results
      print(f"Initial Chat time: {cache_time}s")
      print(f"Completion time with cache (Chat 2): {time_cache_1}s, Response: {out_texts_cache_1[0]}")
      print(f"Completion time with cache (Chat 3): {time_cache_2}s, Response: {out_texts_cache_2[0]}")
      print(f"Completion time without cache (Chat 1+2): {time_no_cache_1}s, Response: {out_texts_no_cache_1[0]}")
      print(f"Completion time without cache (Chat 1+2+3): {time_no_cache_2}s, Response: {out_texts_no_cache_2[0]}")

    if args.task == 'system':
   
      cache_prompt = """Task Description:
          You are an advanced language model trained to understand and classify emotions conveyed in text. Your task is to read short passages and identify the primary emotion expressed from the following options: joy, sadness, anger, fear, surprise, disgust, trust, and anticipation. For each passage, choose the emotion that best matches the overall sentiment of the text.

          Few-Shot Examples:
          Text: "Winning the championship after months of hard work felt incredibly rewarding. The team's joy was palpable, and celebrations lasted throughout the night."
          Emotion: Joy

          Text: "The news of his grandmother's passing hit him hard. He found himself reminiscing about the summer vacations spent at her house, feeling a deep sense of loss."
          Emotion: Sadness

          Text: "She couldn't believe her colleague blatantly lied about her in front of the boss. The injustice of the situation filled her with anger."
          Emotion: Anger

          Text: "Walking home alone, he heard footsteps echoing behind him. Turning around and seeing no one, fear gripped him as he quickened his pace."
          Emotion: Fear

          Text: "The surprise party for her 30th birthday left her speechless. She had no idea her friends and family could pull off something so elaborate without her finding out."
          Emotion: Surprise

          Text: "Finding the rotten vegetables in the fridge, forgotten and hidden behind the milk carton, made her recoil in disgust."
          Emotion: Disgust

          Text: "After years of working together, they had built a foundation of trust. He knew he could count on his team to support him, no matter the challenge ahead."
          Emotion: Trust

          Text: "The night before the launch, she was filled with anticipation. All their planning and hard work were about to be put to the test."
          Emotion: Anticipation

          """ 
      sentiment_prompt_1 = "Text: \"I am very angry at her. I want to pull out her hair\". Emotion:"
      sentiment_prompt_2 = "Text: \"Everytime I think about her it makes me want to cry. I am heartbroken\". Emotion:"

    
    

      # Cache prompt and measure time
      start_time = time.time()
      model.cache_prompt([cache_prompt], cache_id=1)
      cache_time = time.time() - start_time

      # Measure time for completion with caching
      start_time = time.time()
      _, out_texts_cache_1 = model.text_completion([sentiment_prompt_1], max_gen_len=3, cache_id=1)
      time_cache_1 = time.time() - start_time

      start_time = time.time()
      _, _ = model.text_completion([sentiment_prompt_1], max_gen_len=3)
      time_cache_1 = time.time() - start_time
      print(f"This is the generation without the prompt ---> {out_texts_cache_1[0]}")


      start_time = time.time()
      _, out_texts_cache_2 = model.text_completion([sentiment_prompt_2], max_gen_len=3, cache_id=1)
      time_cache_2 = time.time() - start_time

      # Measure time for completion without caching
      start_time = time.time()
      _, out_texts_no_cache_1 = model.text_completion([cache_prompt + sentiment_prompt_1], max_gen_len=3)
      time_no_cache_1 = time.time() - start_time

      start_time = time.time()
      _, out_texts_no_cache_2 = model.text_completion([cache_prompt + sentiment_prompt_2], max_gen_len=3)
      time_no_cache_2 = time.time() - start_time

      # Print results
      print(f"Cache setup time: {cache_time}s")
      print(f"Completion time with cache (Prompt 1): {time_cache_1}s, Response: {out_texts_cache_1[0]}")
      print(f"Completion time with cache (Prompt 2): {time_cache_2}s, Response: {out_texts_cache_2[0]}")
      print(f"Completion time without cache (Prompt 1): {time_no_cache_1}s, Response: {out_texts_no_cache_1[0]}")
      print(f"Completion time without cache (Prompt 2): {time_no_cache_2}s, Response: {out_texts_no_cache_2[0]}")





