# Research results · 25 November 2026

Two kinds of systems were tested: small recurrent policies that operate on structured, authenticated records in a simulator, and a separate next-token SSM trained on ordinary text. Results do not transfer automatically from one to the other.

## 1. Adding memory to a typed policy

Each held history had to complete both steps on the original record and after each of two independent corrections. One seed; 32 held histories.

| Arm | Parameters at end | Updates | Strict held | With added bank disabled |
|---|---:|---:|---:|---:|
| Stay narrow | 60 | 837 | 0/32 | Not applicable |
| Grow at update 93 | 186 | 333 | 32/32 | 0/32 |
| Start wide | 186 | 270 | 32/32 | Not applicable |

All arms used 50,220 scalar parameter-updates, not equal FLOPs or training exposures. At insertion, all 1,280 tested old state coordinates and 256 protected scores were bit-identical; the existing optimizer state was preserved. The narrow arm nevertheless completed 34/96 individual base or correction episodes. All three arms clarified on missing or conflicting records.

## 2. Learning a second typed skill without retraining the first

At each of two seeds (49711 and 71387), both skills were evaluated on the same 24 held histories, with four variants per history and six consecutive correct decisions required. The added skill was 0/24 immediately after insertion and 24/24 after training. The isolated model used no old-task labels after growth.

| Arm | Old skill (each seed) | New skill (each seed) | Exact old-score replay after fitting |
|---|---:|---:|---|
| Isolate the old branch and train the new bank | 24/24 | 24/24 | Yes |
| Tune the grown model freely | 24/24 | 24/24 | No |
| Stay narrow | 24/24 | 0/24 | No |
| Train full-width with both tasks interleaved | 24/24 | 24/24 | No |

Replacing either relevant bank state with another history's state reduced the isolated model's new-skill score to 0/24 on the base variant, at each seed. The typed gate forced clarification when required records were missing or conflicting. The order of viable actions was learned.

An earlier eligibility study helps interpret the gate: a learned three-score head, a factorized learned head, and a hard eligibility rule tied at 48/48 correct on complete cases and 0/96 issued choices on incomplete cases per seed. The two seeds reused six held boards. Only the hard rule excludes unsupported choices by construction under its typed-input assumptions. See the [analysis](ANALYSIS.md#what-the-architecture-guarantees).

## 3. Adding a routed bank to the actual language model

The approximately 10.5M-parameter parent had seen about 65M fresh training tokens across its broad-text lineage (plus replay and document separators). In the frozen 40M-to-65M evaluation, next-token loss fell from 3.173926 to 3.156401 on 31,232 Wikipedia targets and from 5.159307 to 3.974172 on 2,810 writing-guide targets. Neither side was correct on both versions of any of 16 evidence pairs.

Two isolated adaptation arms then trained an additional 220,896 recurrent parameters for 640 updates and 92,160 scored next-token targets each. The mixture used 61,440 prose and 30,720 synthetic targets. The original route bypassed the new bank. All tested original-mode scores and greedy IDs matched at insertion, after fitting, and after a cold replay, on three fixed inputs.

| Held metric | Frozen parent | Prose-only bank | Prose + evidence bank |
|---|---:|---:|---:|
| Public text loss, nats/token | 3.7934 | 3.6673 | 3.7433 |
| Exact answer through EOS | 0/96 | 0/96 | 0/96 |
| Both correct continuations outrank their foils | 0/48 | 0/48 | 0/48 |
| Raw continuation changed when queried object changed | 2/48 | 20/48 | 3/48 |

The prose panel comprises 1,584 targets from three held source groups, not 1,584 independent sources. The language panel comprises 12 generated scenarios crossed with wording, fact changes and query reversal. Neither bank passed the evidence gate; the parent stayed unchanged. An example: swapping the mug and flute between a shed and an attic calls for a different answer. The mixture returned the exact same fragment both times: “In the hall. Where did the bedroom? In”.

## 4. Does freezing the backbone explain the failure?

A separate one-seed comparison gave the frozen-backbone bank and a fully trainable copy 320 updates and the same 12,800 supervised answer tokens each. Neither met the familiar-validation threshold for extending training. Strict success required four complete correct answers as the queried object and its color changed independently.

| Panel | Frozen backbone + bank | Fully trainable copy |
|---|---:|---:|
| Training | 0/6 | 0/6 |
| Familiar wording, new contexts | 0/6 | 0/6 |
| Novel combinations of familiar objects | 0/8 | 0/8 |
| Held-out wording | 0/8 | 0/8 |

Even when directly supplied with the correct color, the full copy answered “It is red.” for both red and blue. Its direct-color diagnostic loss was 0.13879 nats per target, close to the 0.13863 expected if four formatting tokens are predicted perfectly and the balanced color token is guessed. This is a failure below the level of multi-step reasoning, not evidence that longer training could never work. No weights were promoted.

## Audit and reproduction

- [Evidence snapshot](evidence.json) records all 32 typed-plan outcome codes per variant and arm (`C` completion, `D` clarification, `P` wrong first step, `S` wrong second step), per-seed typed-skill metrics, the individual language scoring flags and six representative raw continuations with token IDs. Source artifact hashes identify the local audit inputs; the snapshot is an export, not the complete runner or set of raw logs.
- Run `python reports/2026-11-25/check_evidence.py` from the repository root to recompute headline counts from the released outcome codes and flags. It does not rerun training or check a formal certificate.
- For a fresh from-scratch experiment using this repository's bundled implementation and synthetic fixtures, follow [the original reproduction steps](../../README.md#reproduce). The newer local language bank run cannot be recreated from this snapshot alone: the parent checkpoint and corpus pack are not included.
