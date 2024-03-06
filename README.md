# 🦙🔍 De-redacting Elon's Email with Character-count Constrained Llama2 Decoding

Many are attempting to de-redact Andrej's email, but Elon's email seems [easier to solve](https://twitter.com/khoomeik/status/1765320782777917635) with constrained decoding on a causal LLM due to its much longer prefix.  Below I present a decoding algorithm that searches for completions that meet the **character-count constraints** imposed by the redaction's word breaks.

Here are a few initial completions it generated that match the required word lengths and may be worth exploring (though it wasn't able to complete these):
- "`Nothing matters but the scale and scale requires money`..."
- "`Nothing changes at all unless you raise several`..."
- "`Google research is far beyond any other research group`..."
- "`Google Research is now doing the most cutting edge work`..."
- "`Please consider how to change the above scenario from zero probability`..."
- "`Please consider how to ensure that your research team and technology remain aligned with`..."
- "`Please consider how to raise and spend capital with the appropriate degree`..."
- "`Please consider all the facts and think through your plan thoroughly before making any decision`..."

Some notes about the algorithm:
- we select the top `top_k_to_decode` logits to check if they correspond to tokens with correct word length
- we look for tokens with a character count that matches `word_length` or `word_length - 1` to account for a space or punctuation
- amongst these tokens with the correct length, we then select those with at least `min_p_logits * i * prob_gamma`
  - the `i * prob_gamma` term increases minimum probability as the sequence gets longer (since possibilities will naturally narrow)
- for each next token prediction, we recursively continue our decoding process

TODO:
- [ ] handle subword tokens
- [ ] convert from recursive to loop + queue
- [ ] parallelize/batchify the search process
- [ ] tune the search constraint hyperparams
- [ ] try a larger model or something fine-tuned on Elon's corpus
