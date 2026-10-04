`current_memloop_parent.tar.gz` is the unmodified `qwen_latent_cot` tree produced
by `git archive` from commit `6f936b7068272b999c0a9b7ec0a1434dbcd9161e`.
It contains source only, with no model weights. The test extracts it into a
temporary directory and imports it in an independent Python subprocess.

`run_current_memloop_parent.py` calls the parent's `_forward_flow_loop`, using
native weights, noise, timestep, and prompt cache tensors supplied by the test.
Memory initialization uses the original SOI/EOI embeddings and seed 0.
The compatibility path never serves as its own test oracle.
