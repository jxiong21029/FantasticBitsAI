Here, I was working on initializing, or pretraining, the policy via behavioral cloning.
When fine-tuning the policy downstream using RL, a common practice is to also pretrain
the value function via standard policy evaluation (with a frozen policy).

However, this becomes nontrivial if parameters are shared between the policy and value
functions, which is common practice for *tabula rasa* RL. I took the straightforward
approach of simply constraining the policy to be close to the original policy via a KL
loss. However, I noticed during early testing runs that this process actually
*improved* the performance of the policy. I hypothesized that training the encoder
with a value approximation objective improved robustness akin to a self-supervised
loss. A similar idea exists in PPG.

The results from further, more rigorous testing are below. Parameters were tuned using
my personal interval halving search.

| Description              | Performance (goals / episode) |
|--------------------------|-------------------------------|
| Policy after BC          | 2.554 (2.533, 2.590)          |
| Policy after VF training | 2.581 (2.522, 2.684)          |
| Improvement              | +0.030 (-0.066, +0.118)       |

Policies are evaluated over 200 episodes. Results report IQM over 5 runs w/ 95%
bootstrap confidence intervals. (Improvement is calculated per-run before aggregation).

Therefore, training with the value objective in this fashion has no statistically
significant impact on the performance of the policy. However, this lack of significance
may be simply a result of the relative low number of trials. I expect, based on this
data, that the value function objective likely improves performance, but not by a large
margin; and any difference, if indeed present, becomes less pronounced as the amount of
demonstration data increases.

| Hyperparameter       | Value |
|----------------------|-------|
| learning rate (BC)   | 1e-3  |
| weight decay (BC)    | 1e-4  |
| learning rate (VF)   | 1e-3  |
| learning rate (VF)   | 1e-3  |
| KL penalty beta (VF) | 1     |
