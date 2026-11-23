# Growing a language model under a decision guarantee

Can a language model gain useful reasoning capacity without gaining the ability to approve an unsupported plan? This project trains a small autoregressive state-space model from scratch. It predicts every next token, consumes a prompt in one pass, then generates words one at a time from its recurrent state. Its vocabulary includes two reserved decision tokens, `<confirm>` and `<defer>`. Both compete directly with all ordinary words.

## The experiment

Two shipments have different permits, capacity estimates and load estimates. Suppose A has at least 500,000 units of capacity and needs at most 400,000. B has at least 400,000 and may need 500,000. Both have permits. Switching the requested shipment from A to B should change the answer even though every receipt stays the same. Guarantees refer to conservative bounds on the underlying facts; the model's ability to write a useful route and explanation is measured separately.

* **Learned state.** A token selects per-coordinate retention and new input. Separate recurrent blocks learn features for next-word prediction. Ordinary text influences those features and the full-vocabulary competition. Sources: [model](model.py), [constrained recurrence and readout](PROOF.md).
* **Protected state.** A parallel part of the same recurrent model stores trusted receipts. The simplest experiment stores two prerequisite flags. The interval experiment stores three primitive values per named plan and computes feasibility inside the recurrence. Words such as “approved” in an untrusted note cannot replace a receipt. See [interval model](interval_model.py) and [data contract](DATA.md).
* **What is certified.** In the specified greedy decoding protocol, a missing permit, missing bound, or insufficient guaranteed capacity keeps the raw `<confirm>` score below `<defer>`. Every learnable weight setting admitted by the model obeys that separation, including immediately after initialization, after optimization, and after width growth. Sampled text, the truth of a forged receipt, and the meaning of an ordinary sentence fall outside this property. The proof and its numerical assumptions are in [PROOF.md](PROOF.md).
* **What must be learned.** The model still has to emit a decision at the right time, remember the requested route, bind the correct entity, and explain the choice in words. It can satisfy the guarantee while failing every one of these tasks. Score the exact raw continuation and positive coverage alongside the guaranteed inequality.

## Growth and controls

Growth appends an independently driven block with a zero output head. Its first gradient can open a path to the vocabulary without changing any existing logit at insertion. The experiment compares all vocabulary logits before/after insertion and retains the original Adam moments. It measures loss with the trained new head removed. Distinguish deliberate test insertion (`--growth-mode smoke`) from the plateau arm, which requires repeated development observations of both loss and useful-response performance. A brief calibration cannot establish saturation.

Use the same ordered samples and seed to compare small continuation, same-size optimizer restart, and an identical final-size model trained from scratch. Report token count, parameter-target products, wall time, steps at full size and held-out performance. Parameter-target products are an accounting proxy. Counting equal updates alone gives a bigger model more compute. Growth earns another rung only if it retains the guarantee and improves a predeclared capability metric relative to a fair control.

## Reproduce

Python 3.12 with PyTorch, no network access needed for the included data. The Windows supervisor uses the Omen's existing pinned Python 3.12.10 / Torch 2.14.1+cu130 installation. Use the actual paths for that installation as environment variables `PY312` and `TORCH_SITE`. Create a new output directory outside the checkout for the campaign. The supervisor admits one child per invocation; run these sequentially on a single GPU.

~~~powershell
$campaign = Join-Path (Split-Path (Get-Location)) 'my-ssm-runs'
New-Item -ItemType Directory -Path $campaign -Force | Out-Null
& $env:PY312 -I -S -B launch.py --site $env:TORCH_SITE --mode test --receipt "$campaign/tests.json"
& $env:PY312 -I -S -B launch.py --site $env:TORCH_SITE --mode train --receipt "$campaign/base-supervisor.json" -- --device cuda --task interval --output "$campaign/base" --arm base --width 16 --seed 12 --batch-size 12 --steps 128 --eval-every 64 --max-new-tokens 24 --max-seconds 100
~~~

Use `experiment.py --help` for controls and resume arguments. Each resume consumes a matching source, configuration and SHA-256-pinned local checkpoint. Use new output and receipt paths for each run. Complete local artifacts include the learned weights, optimizer state, runtime measurements and per-example outputs. Only selected reproducible source and sanitized evidence belong in Git. The open research questions and prior work are in [RESEARCH.md](RESEARCH.md).
