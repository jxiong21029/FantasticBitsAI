Here, I was working on initializing, or pretraining, the policy via behavioral cloning.
When fine-tuning the policy downstream using RL, a common practice is to also pretrain
the value function via standard policy evaluation (interacting with the environment
using a frozen policy).

However, this becomes nontrivial if parameters are shared between the policy and value
functions, which is common practice for *tabula rasa* RL. I took the straightforward
approach of simply constraining the policy to be close to the original policy via a KL
loss. However, I noticed during early testing runs that this process actually
*improved* the performance of the policy. I hypothesized that training the encoder
with a value approximation objective improved robustness akin to a self-supervised
loss. A similar idea exists in PPG.

Results from further, more rigorous testing are below.

| Description              | Performance (goals / episode) |
|--------------------------|-------------------------------|
| Policy after BC          | 2.603 (2.541, 2.657)          |
| Policy after VF training | 2.599 (2.552, 2.645)          |
| Improvement              | -0.005 (-0.047, +0.045)       |

Policies are evaluated over 200 episodes. Results report IQM over 20 runs w/ 95%
bootstrap confidence intervals. (Improvement is calculated per-run before aggregation).

Therefore, pretraining with the value objective in this fashion has no statistically
significant impact on the performance of the policy.

Parameters were tuned using my personal interval halving search algorithm, up to a
maximum of two steps per decade.

| Hyperparameter       | Value |
|----------------------|-------|
| learning rate (BC)   | 1e-3  |
| weight decay (BC)    | 1e-4  |
| learning rate (VF)   | 1e-3  |
| weight decay (VF)    | 1e-3  |
| KL penalty beta (VF) | 1     |
