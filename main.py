from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", load_in_4bit=True)
model.eval()

import torch

word_lengths = [7,8,3,3,6,4,5,8,5,4,11,7,7,4,8,7,4,4,4,3,6,4,4,9,9,5,5,4,6]

email = """From:  Elon Musk
To:  Ilya Sutskever, Greg Brockman
CC:  Sam Altman, Andrej Karpathy
Date: Wed, Dec 26, 2018 at 12:07 PM
Subject: I feel I should reiterate
My probability assessment of OpenAI being relevant to DeepMind/Google without a dramatic change in execution and resources is 0%. Not 1%. I wish it were otherwise.

Even raising several hundred million won't be enough. This needs billions per year immediately or forget it.

Unfortunately, humanity's future is in the hands of Demis.

https://deepmind.google/blog/alphafold/

And they are doing a lot more than this.

"""
# NOTE: https://twitter.com/khoomeik/status/1765357875663454460 for URL de-redaction hypothesis

top_k_to_decode = 200
min_p_logits = 0.004
prob_gamma = 1.3
tok_seq = tokenizer(email, return_tensors='pt')['input_ids'].to('cuda')

def dfs_decode(tok_seq, wl_idx):
  if wl_idx == len(word_lengths):
    return []

  prev_completion = tokenizer.decode(tok_seq[0, 180:])

  with torch.no_grad():
    output = model(tok_seq)
  probabilities = torch.softmax(output.logits[0, -1], dim=0)
  top_tok_indvals = torch.topk(probabilities, top_k_to_decode)
  top_toks = tokenizer.batch_decode(top_tok_indvals.indices)

  del output
  del probabilities

  next_toks = []
  for i, tok in enumerate(top_toks):
    if len(tok) in (word_lengths[wl_idx], word_lengths[wl_idx]-1) and top_tok_indvals.values[i] > min_p_logits * wl_idx * prob_gamma:
      next_toks.append(tok)

  if len(next_toks) == 0:
    print('None')
    return None
  else:
    next_tok_ids = tokenizer(next_toks, add_special_tokens=False)['input_ids']
    for i, next_tok_id in enumerate(next_tok_ids):
      print(prev_completion, next_toks[i])
      new_tok_seq = torch.IntTensor([tok_seq.squeeze().tolist() + next_tok_id])
      completions = dfs_decode(new_tok_seq, wl_idx + 1)

      if completions is not None:
        for completion in completions:
          completions.append([next_tok_id] + completion)

    return completions

torch.cuda.empty_cache()
tok_seq = dfs_decode(tok_seq, 0)
