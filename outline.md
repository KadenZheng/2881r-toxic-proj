# Reproducing SparC³ Circuit Discovery & Correction

# on LLaMA‑

To replicate the **SparC³** framework’s circuit discovery and targeted model correction on a LLaMA‑3 model
with minimal custom engineering, you can leverage a combination of open-source implementations and
best practices. The SparC³ approach uses **Layer-Wise Relevance Propagation (LRP)** – particularly an
attention-aware variant ( _AttnLRP_ ) – to attribute model predictions to internal components, then prunes low-
relevance components or those implicated in undesirable behaviors. Below, we outline key resources, tools,
and guidelines:

## Existing Implementations of SparC³ and AttnLRP

```
Official SparC³ Repository: The authors of SparC³ have a GitHub repo, erfanhatefi/SparC3 , which will
host the official implementation. As of now it’s marked “work in progress,” but it confirms the
methodology (attribution-guided unstructured pruning for compression, circuit extraction, and
targeted correction). Keep an eye on this repo for ready-to-use code aligning exactly with the paper’s
methods.
```
```
AttnLRP / LRP for Transformers: SparC³ builds on AttnLRP , an LRP extension tailored for
Transformer models. Fortunately, the authors have released an open-source toolkit called LXT
(Layer-wise Relevance Propagation eXplains Transformers). LXT implements AttnLRP for large
language models and vision transformers, and is available via pip (pip install lxt). It
supports LLaMA-2 and LLaMA-3 models out-of-the-box. This library will allow you to compute
per-weight, per-neuron, or per-head attributions with minimal code. For example, LXT’s
documentation provides a quickstart showing how to patch a HuggingFace model and obtain
relevance scores in a single backward pass. Using LXT means you don’t need to re-implement
LRP; you can directly apply AttnLRP to LLaMA-3 to get the importance scores needed for pruning.
```
```
Related Attribution-Pruning Pipelines: In addition to SparC³’s code, there are other projects that
align with attribution-based pruning:
```
```
Wanda – Pruning by Weights and Activations: Wanda is a simple one-shot pruning method
introduced by Sun et al. (2023) and used as a baseline in SparC³. Its official PyTorch implementation
is available (locuslab/wanda). Wanda computes importance as the product of weight
magnitude and input activation norm, using only a forward pass (no gradients). While less fine-
grained than LRP (Wanda can’t attribute importance to individual attention heads or neurons ),
it’s very easy to use and has been tested on LLaMA models. The repository supports multiple
methods (magnitude pruning, Wanda, SparseGPT) and provides command-line scripts for pruning
HuggingFace LLaMA checkpoints. For example, to prune LLaMA-7B to 50% sparsity with Wanda
unstructured, you can simply run:
python main.py --model decapoda-research/llama-7b-hf --prune_method wanda --
sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/
```
### •

```
1 2
```
### •

```
3
```
```
4
5
```
```
6 7
```
### •

### •

```
8 9
10
10
11
```

```
unstructured/wanda/. Adjusting --sparsity_type allows structured N:M sparsity (e.g.
--sparsity_type 2:4 for 2:4 semi-structured pruning). This repository can serve as a
template – you could plug in LRP-based scores in place of Wanda’s if needed, but even out-of-the-box
it’s a minimal-effort way to prune LLaMA models.
Safety Alignment Pruning (Wei et al. 2024): Another relevant codebase is alignment-attribution by
Boyi Wei et al. (ICML 2024). This project uses pruning to analyze safety alignment brittleness. It
provides scripts to identify “safety-critical” neurons by comparing attributions on aligned vs. base
datasets. The repository supports LLaMA-2 (7B/13B chat models) and implements both “top-down”
pruning (pruning most harmful neurons first, by adding --neg_prune) and set-difference pruning
(pruning neurons that are important to safety vs utility to isolate alignment-specific circuits).
For example, they let you specify a --prune_data (such as align for a dataset of safety
prompts) and a --prune_method (e.g. wanda or wandg for their variant of SNIP) to remove a
fraction of neurons with one command. This codebase serves as a practical guide to
implementing targeted pruning for behavior correction: you can study its usage of reference
datasets and scoring methods, and even adapt it to use LRP scores (by dumping LRP attributions as
“importance” and pruning accordingly). It’s a valuable example of a lightweight pipeline for targeted
neuron-level pruning without extensive re-engineering.
```
## LRP Toolkits for LLaMA-3 (AttnLRP)

For conducting Layer-wise Relevance Propagation on LLaMA models, the **LXT library** mentioned above is
the top recommendation. It was developed by the same research group as SparC³ and AttnLRP, ensuring
methodological fidelity. Key points for using LXT with LLaMA-3:

```
Supported Models: LXT explicitly lists support for LLaMA 2/3 models (as well as other
transformer families like Qwen, BERT, GPT-2, etc.), meaning it has built-in model wrappers or configs
for those architectures. This greatly simplifies integration, as you won’t need to write custom
backward rules for LLaMA’s layers – LXT has done that (including the attention mechanisms, via
AttnLRP).
```
```
Installation and Use: After installing via pip, you can apply LXT in two ways. The “efficient” mode
monkey-patches the model to use a fast gradient-based approximation of LRP – essentially
computing $input \times gradient$ as the relevance (this was shown to match decomposition
methods for transformers ). This mode is recommended for speed and ease. Alternatively, the
“explicit” mode in LXT implements the exact LRP rules with custom autograd functions for each
operation ; it’s slower but useful if you want to verify the LRP math. In practice, the efficient mode
should suffice for guiding pruning.
```
```
Documentation & Examples: The LXT documentation provides a quickstart tutorial using LLaMA as
an example. It guides you through patching a Hugging Face LLaMA model and performing a
forward/backward pass to get relevance scores at any layer. Example scripts are also included in the
repo (examples/ directory) for common tasks. Using these, you can quickly obtain:
```
```
Parameter-level attributions: e.g. relevance of each weight in a linear layer.
Neuron-level attributions: e.g. aggregate relevance of each hidden unit (by summing its incoming/
outgoing weight relevances).
```
```
12
13
```
### •

```
14
```
```
15 16
```
```
17 18
```
### •^19

### •

```
6 7
```
```
20
```
```
21
```
### •

```
22
```
### •

### •


```
Head-level attributions: AttnLRP provides a decomposition of attention scores by head , so you can
identify which attention heads are most responsible for a given output. This is crucial if you plan
structured pruning of whole heads – something standard gradient methods or Wanda can’t directly
pinpoint.
```
In short, **LXT is the go-to tool** for faithful attribution on LLaMA-3. It handles the heavy lifting of AttnLRP,
enabling you to focus on using the output scores for pruning decisions.

## Constructing Reference Sets for Behavior Discovery

A core idea in SparC³ is using different **reference input sets** to target specific behaviors. The choice of
reference data is critical: as the paper notes, the set of examples you use to compute attribution scores has
a strong influence on which parameters appear “important”. Here are best practices for assembling
these sets, especially for undesirable behaviors (toxicity, repetition, bias, etc.):

```
General Reference Set (Baseline): First, prepare a set of inputs that represent the model’s general,
desirable behavior. In SparC³, they sample from a broad corpus like C4 or Wikipedia text. These
should be diverse, neutral prompts that cover the model’s normal range of content. This set (call it
R <sub>gen</sub>) is used to identify components important for the model’s overall capabilities. For
example, SparC³ draws 128 sequences of length 2048 from C4 (with different random seeds) to serve
as reference for Llama models. Using a reasonably large and varied baseline set ensures that
when you later remove components, you don’t inadvertently damage general language ability.
```
```
Behavior-Specific Set: Next, curate a set of inputs that elicit the target behavior you want to
analyze or mitigate. This could be an undesirable trait like toxicity, a specific task (for circuit
discovery), or any narrow behavior. In the paper’s model correction experiments, they tackled two
behaviors:
```
```
Toxic outputs: They leveraged the RealToxicityPrompts dataset, which provides prompts known to
trigger toxic or hateful completions. From this, they selected 93 prompts that led to highly toxic
responses from the model (using Jigsaw’s Perspective API scores to quantify toxicity). This
yielded a focused set R <sub>tox</sub> capturing the toxicity-triggering situations.
Repetitive outputs: Because a standard dataset for repetition may not exist, they generated prompts
likely to induce repetition. Specifically, the authors asked ChatGPT to produce prompts that cause a
model to respond with repetitive text. From these, they collected 53 prompts that
consistently made the model repeat itself (measured via a “Response Uniqueness Ratio”, which is low
when a response is repetitive). This became R <sub>rep</sub>.
```
For your use case, identify or create a dataset _R_ <sub>behav</sub> for the behavior of interest. If the
behavior is “similar to but distinct from toxicity,” for example, it might be _biased or harmful content of a
specific kind_. You could use existing resources (e.g. prompts triggering political bias, or a subset of a toxicity
dataset focusing on a particular category), or generate prompts with an instructive LLM as the authors did.
**Crucially, ensure that these prompts reliably elicit the behavior** (so that the model components
engaged are truly the ones relevant to that behavior).

```
Isolation of Behavior vs General Ability: The power of SparC³’s method comes from comparing
attributions on the behavior-specific set to those on the general set. The paper explicitly constructs a
```
### •^23

```
10
```
```
24
```
### •

```
11
```
```
11
```
### •

### •

```
25
25
```
### •

```
26 27
```
```
28
```
### •


```
differential attribution by subtracting the general importance from the behavior-specific
importance. Intuitively, you want components that are highly relevant to the unwanted behavior
and minimally important to normal tasks. Thus, try to make R <sub>behav</sub> as targeted as
possible, and R <sub>gen</sub> as representative as possible of normal model use. The authors
emphasize that reference samples must “accurately isolate the behavior of interest without
overlapping with general capabilities”. In practice, this might mean excluding prompts that
overlap in content or style between the two sets. (For example, if you were targeting political bias ,
R <sub>behav</sub> might consist of politically charged questions, while R <sub>gen</sub> is neutral
Wikipedia sentences; you’d avoid having political content in the general set to keep the signals
separate.)
```
```
Size of Reference Sets: Using more reference samples can improve attribution fidelity for complex
behaviors, but it comes at higher computation cost (LRP requires a forward and backward pass per
sample). SparC³ found that 128 samples was a good trade-off for LLM pruning , and they
noted diminishing returns beyond a few hundred samples due to GPU memory limits and the nature
of Wanda vs LRP. If resources are abundant (as in your case on a GPU cluster), you can
experiment with larger sets for more stable results. Start with on the order of 100 samples for each
set and increase if needed – keeping in mind that Wanda attributions are cheaper (forward-only)
than LRP (forward+backward).
```
In summary, **prepare two datasets** : one general-purpose and one behavior-specific. Use the same
methodology (forward passes with LXT or Wanda) on both to get attribution scores, then compare or
subtract them to find the “circuit” – the parameters uniquely supporting the specific behavior. This approach
follows the SparC³ model correction technique of focusing removal on the intersection of _high relevance to
bad behavior_ and _low relevance to normal tasks_.

## Pruning Workflow: From Attribution to Model Correction

With attribution scores in hand (via LXT or another method) and well-chosen reference sets, the pruning
and correction process can be implemented using existing tools with minimal new code:

```
Ranking Parameters by Attribution: First, decide the granularity at which you will prune. If you aim
to remove fine-grained “edges” (individual weight elements) as in unstructured pruning, you can
rank all weights by their relevance score. If instead you plan structured pruning (entire neurons or
heads), aggregate the scores per neuron or per head (e.g. sum of absolute relevances of all weights
feeding into a neuron). In SparC³’s circuit discovery, they evaluated both levels: removing whole
neurons or heads (structured) vs. individual weights (unstructured). The AttnLRP/LXT toolkit
makes it possible to get scores at either level. For instance, it decomposes attention scores by head
, allowing you to assign a relevance value to each head.
```
```
Pruning with Existing Tools: To actually zero-out or remove the identified components, leverage
frameworks:
```
```
Unstructured Pruning: The Wanda code can be co-opted here. It already supports global or layer-wise
unstructured pruning by importance values. You could modify Wanda’s scoring function to use
LRP scores instead of its default (weight*activation) if needed. However, given that SparC³ found
Wanda’s attributions nearly as effective for pure compression , you might first try using Wanda
```
```
29
```
```
27
```
### •

```
30 31
```
```
32
```
```
33
```
```
34 35
```
### •

```
36 37
```
```
23
```
### •

### •

```
38
```
```
39
```

```
as-is for unstructured pruning (it requires only forward passes on reference data). Wanda’s scripts
will handle applying the mask and saving the pruned model. Simply point --prune_method to
wanda or sparsegpt as desired, and set --sparsity_ratio to your target (e.g. 0.5 for 50%
weights removed). The tool will output a pruned model checkpoint which you can then evaluate
or fine-tune if needed.
```
```
Structured Pruning: For larger structural units like neurons or heads, you may use Hugging Face
Transformers utilities or manual masking. For example, the Transformers library has a built-in
prune_heads method for some models (e.g. BERT) to remove attention heads. LLaMA models
don’t have a one-call head prune function in the library as of now, but you can implement it by
setting the projection matrices of the target heads to zero and tweaking the model’s forward pass to
skip them. A simpler approach is masking : for a neuron, zero out all incoming and outgoing weights
(essentially removing its influence). For an attention head, zero out its output projection and input
weights. This “virtual removal” is what SparC³ did in experiments: e.g. to remove 100 neurons from
LLaMA’s MLP, they zeroed those neurons’ weights and showed toxicity dropped without hurting
perplexity. If you prefer a library solution, you can use PyTorch’s pruning API
(torch.nn.utils.prune). It allows structured pruning by specifying entire channels (filters) to
prune. For instance, to prune neurons in a linear layer you could prune out entire rows or columns of
the weight matrix. Keep in mind that a neuron in a transformer MLP spans two weight matrices (the
input and output of that feed-forward layer), so true removal means masking both.
```
```
Example – Targeted Circuit Removal: To illustrate a minimal workflow, consider toxic response
suppression :
```
```
Use LXT to compute attribution scores for all neurons in the feed-forward layers on R <sub>tox</sub>
(toxic prompts) and on R <sub>gen</sub> (general prompts).
Compute a “score” for each neuron = (LRP relevance on toxic set) – (LRP relevance on general set)
(or another combination that highlights toxic-specific importance).
Rank neurons by this score descending. The highest scores indicate neurons most uniquely
contributing to toxicity.
Remove the top k neurons. You could start with a small k (e.g. 100 neurons as the paper did ) by
zeroing them out or using a pruning mask.
Evaluate the model’s toxicity (e.g. with Perspective API or a toxicity classifier) and general
performance (perplexity, zero-shot accuracy) before and after. SparC³ reported that pruning just 100
neurons from LLaMA’s FC1 layers via LRP significantly lowered toxic output while not degrading
WikiText perplexity.
```
You can follow a similar procedure for any behavior: identify the circuit with attributions, then ablate it in a
targeted way. The alignment-attribution codebase is a great reference here – for example, they implement a
“safety-first pruning” which corresponds to removing neurons most important for safety (alignment) while
preserving utility. In their README, they show how to do this via a single script invocation, which
you could mimic for your needs.

```
Lightweight Examples & Notebooks: While there may not be an exact “SparC³ tutorial” notebook
published yet, the combination of the above tools covers it. For a hands-on starting point:
Try running LXT’s example to get relevance scores for a single input on LLaMA. This will ensure your
environment (transformers, model weights, etc.) is set up correctly for LRP.
```
```
12
```
### •

```
40 41
```
### •

### •

### •^29

### •

### •^42

### •

```
40 43
```
```
44 45
```
### •

### •


```
Next, use Wanda’s script on a small sparsity (say 10%) to prune LLaMA-3, just to see the flow of
loading a model, pruning, and saving.
Then, integrate them: for instance, you might dump LRP scores to a file and then load them in a
modified version of Wanda’s main.py to decide which weights to prune. Since Wanda’s code is
quite compact and already handles model loading and saving, this could save engineering effort.
Alternatively, if focusing on neuron or head removal, you can write a short Python snippet using the
transformers library: iterate over identified bad neurons and zero their weights, then save the
state dict. This can be done in a few dozen lines and avoids needing a custom model class.
```
The key is that **most heavy lifting (attribution, masking, evaluation)** is provided by existing libraries, so
you can avoid writing complex model logic. Use these tools in sequence: _AttnLRP for scoring → Pruning
mechanism (Wanda or PyTorch) for removal → Evaluate → Iterate_.

## Structured vs. Unstructured Pruning Considerations

Choosing between structured and unstructured pruning will affect both the engineering complexity and the
results:

```
Unstructured Pruning: This removes individual weights regardless of their location. SparC³
primarily uses unstructured pruning for compression because it achieves higher sparsity and finer-
grained circuit discovery. The advantage is you can prune very aggressively (e.g. 50%+
weights) and zero out the truly least important connections across the model. Tools like Wanda and
SparseGPT excel at unstructured pruning – they’ve shown they can prune 50% of LLaMA weights
with minimal perplexity hit. The downside is that unstructured sparsity leads to irregular
weight matrices that current hardware can’t accelerate well (no speed-up, just memory savings,
unless you use specialized sparse runtime). From an implementation view, unstructured pruning is
easy: generate a mask for each weight matrix and apply it. Recommendation: If your main goal is
analyzing circuits or simply reducing model size, unstructured is fine. Use Wanda’s global pruning or
PyTorch’s prune.global_unstructured with a mask of lowest-LRP weights.
```
```
Structured Pruning: This removes whole structural units (neurons, heads, or even entire layers). It’s
more interpretable in many cases – e.g. “these 2 attention heads were the toxic circuit” is easier to
reason about than “these 7000 weights spread across the network”. It also yields models that might
run faster (e.g. if entire heads are pruned, you can modify the model to skip those computations).
SparC³ demonstrated that in some contexts (like the IOI task circuit), structured pruning can be
effective, though the circuits found were a bit less sparse than with unstructured methods.
Best practices for structured pruning:
```
```
Attention Heads: Use AttnLRP to get per-head relevance. Then you can remove the lowest-
ranked heads. In prior work on smaller models, pruning ~20-40% of heads often has little
performance impact (e.g. see Michel et al. 2019). For LLaMA, you may want to test removing a few
heads at a time and evaluating, as large cuts could degrade quality. If using Hugging Face models,
implement a custom forward that skips pruned heads or set their weights to zero. (Note: Wanda
cannot directly score heads, since it works at weight level and cannot attribute multi-weight components
like an entire head’s block of parameters .)
MLP Neurons: Compute neuron importances as described, and remove the least important
neurons. You can zero out the corresponding row in the second projection (W₂) and column in the
```
### •

### •

### •

### •

```
46 41
```
```
39 9
```
### •

```
47 48
```
### •^23

```
49
```
-


```
first (W₁) to effectively remove a neuron’s contribution. The SparC³ results indicate that only a small
fraction of neurons are responsible for certain behaviors (100 neurons pruned out of ~ in a 6.7B
model to fix toxicity) , so structured pruning at the neuron level can be a very targeted fix.
Again, PyTorch’s structured pruning utility can prune out entire channels in linear layers by L1 norm
```
- you might feed it a custom importance dict based on LRP scores to automate this.

```
Layers or Blocks: SparC³ did not remove entire layers, but this is another structured approach (less
relevant for fine-grained behavior editing, more for compression). There are known methods
(LayerDrop, etc.) for dropping transformer layers, but those typically require fine-tuning after
removal. Given you have LLM-specific methods, it’s better to focus on heads and neurons where you
can precisely pinpoint bad actors.
```
```
Hybrid Approaches: You can combine structured and unstructured techniques. For example, you
might prune a few whole heads that are malicious, and also prune a percentage of weights across
the remaining network for extra compression. The SparC³ paper suggests that the optimal
granularity might differ by context and encourages exploring hybrids guided by attribution.
Since you want to stay true to their methodology: start with their approach (row-wise unstructured
for general pruning , and neuron-level for targeted deletion ) and then adjust if needed.
Always monitor the model’s core performance when you do structured prunes; it’s easier to
accidentally remove something important when you take out a whole neuron or head. A good
practice is to plot a performance vs sparsity curve as SparC³ did – if you see a sharp drop at a
certain point, you know that component was critical.
```
**Recommended tools for structured vs unstructured:** The **Wanda code** covers unstructured and N:M
semi-structured sparsity (which is a specific hardware-friendly pattern) out of the box. For fully
structured (heads/neurons), no off-the-shelf tool will do it automatically for LLaMA, but writing a small
routine or using the Hugging Face model’s own methods is feasible. If you prefer not to modify the model
code, an alternative is to perform structured pruning by fine-grained masking: e.g. to prune a head, mask
all 64 (for example) weight parameters in its output projection as zero – effectively the same as removing it.
The resulting model will behave the same as if the head weren’t there (though it will still compute it, just
with zero output).

**In summary** , to minimize engineering effort, you can compose the solution from these building blocks: use
**LXT/AttnLRP for attribution** on carefully chosen reference sets, utilize **Wanda or similar pruning scripts**
to apply the pruning (possibly with slight modifications to use custom scores), and follow the **SparC³
guidelines for circuit isolation** (prune the intersection of “important for bad behavior” and “unimportant
for good behavior” components ). All the recommended tools (AttnLRP via LXT, Wanda, the alignment
pruning code) are publicly available and have documentation or examples to get you started quickly. By
using these, you can largely avoid writing new low-level code, instead focusing on assembling the pipeline
and verifying that the model’s performance and behavior align with expectations after each pruning
intervention. Good luck with your implementation – with the resources above, you should be able to
reproduce the key results of SparC³ on LLaMA‑3 while keeping the engineering overhead to a minimum.

```
40 41
```
### •

### •

```
50
```
```
38 40
```
```
51
```
```
52
```
```
34
```

**Sources:**

```
Hatefi et al. , “ Attribution-guided Pruning for Compression, Circuit Discovery, and Targeted Correction in
LLMs ,” 2025
Achtibat et al. , “ AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers ,” ICML 2024
```
- _LXT code repository_
Sun _et al._ , “ _A Simple and Effective Pruning Approach for Large Language Models (Wanda)_ ,” 2023 – _Wanda
code_
Wei _et al._ , “ _Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications_ ,” ICML
2024 – _Alignment pruning code_
SparC³ Appendix & Experimental Details – reference set design and pruning results

GitHub - erfanhatefi/SparC3: Attribution-guided Pruning for Compression, Circuit Discovery, and
Targeted Correction in LLMs
https://github.com/erfanhatefi/sparc

Attribution-guided Pruning for Compression, Circuit Discovery, and Targeted Correction in LLMs
https://arxiv.org/html/2506.13727v

GitHub - rachtibat/LRP-eXplains-Transformers: Layer-wise Relevance
Propagation for Large Language Models and Vision Transformers [ICML 2024]
https://github.com/rachtibat/LRP-eXplains-Transformers

GitHub - locuslab/wanda: A simple and effective LLM pruning approach.
https://github.com/locuslab/wanda

GitHub - boyiwei/alignment-attribution-code: [ICML 2024] Assessing the
Brittleness of Safety Alignment via Pruning and Low-Rank Modifications
https://github.com/boyiwei/alignment-attribution-code
