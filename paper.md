# arXiv:2506.13727v1 [cs.LG] 16 Jun 2025

## Attribution-guided Pruning for Compression, Circuit

## Discovery, and Targeted Correction in LLMs

```
Sayed Mohammad Vakilzadeh Hatefi^1 Maximilian Dreyer^1 Reduan Achtibat^1
```
```
Patrick Kahardipraja^1 Thomas Wiegand^1 ,^2 ,^3 Wojciech Samek^1 ,^2 ,^3 ,†
```
```
Sebastian Lapuschkin^1 ,^4 ,†
```
(^1) Department of Artificial Intelligence, Fraunhofer Heinrich-Hertz-Institute
(^2) Department of Electrical Engineering and Computer Science, Technische Universität Berlin
(^3) BIFOLD - Berlin Institute for the Foundations of Learning and Data
(^4) Centre of eXplainable Artificial Intelligence, Technological University Dublin
†corresponding authors:{wojciech.samek,sebastian.lapuschkin}@hhi.fraunhofer.de

## Abstract

```
Large Language Models (LLMs) are central to many contemporary AI applications,
yet their extensive parameter counts pose significant challenges for deployment in
memory- and compute-constrained environments. Recent works in eXplainable AI
(XAI), particularly on attribution methods, suggest that interpretability can also
enable model compression by identifying and removing components irrelevant to
inference. In this paper, we leverage Layer-wise Relevance Propagation (LRP)
to perform attribution-guided pruning of LLMs. While LRP has shown promise
in structured pruning for vision models, we extend it to unstructured pruning in
LLMs and demonstrate that it can substantially reduce model size with minimal
performance loss. Our method is especially effective in extracting task-relevant
subgraphs – so-called “circuits” – which can represent core functions (e.g., indirect
object identification). Building on this, we introduce a technique for model correc-
tion, by selectively removing circuits responsible for spurious behaviors (e.g., toxic
outputs). All in all, we gather these techniques as a uniform holistic framework
and showcase its effectiveness and limitations through extensive experiments for
compression, circuit discovery and model correction on Llama and OPT models,
highlighting its potential for improving both model efficiency and safety. Our code
is publicly available athttps://github.com/erfanhatefi/SparC3.
```
## 1 Introduction

```
Since the introduction of the Transformer architecture [ 36 ], language modeling has undergone a
paradigm shift, enabling the development of models with unprecedented scale and performance.
However, the resulting Large Language Models (LLMs), often comprising hundreds of billions
of parameters, pose significant challenges in terms of training efficiency, storage requirements,
and inference cost. For example, storing model weights alone can require hundreds of gigabytes
of memory, not accounting for the additional overhead during training and deployment. These
limitations are especially pronounced when LLMs are applied to narrower tasks than those they were
originally trained for, motivating the need for model compression and adaptation. Moreover, when
models exhibit undesirable behaviors, retraining or even fine-tuning can be prohibitively expensive –
highlighting the need for efficient, post-hoc tools for model inspection and targeted modification.
```
```
Preprint. Under review.
```

Figure 1: Overview over the three core applications of our attribution-based pruning framework.a)
Attributing model components on a set of reference samples that are drawn from a general-purpose
corpus (e.g.,WikiText2 or C4 [ 20 , 25 ]) enables effective model compression when pruning the
components identified as least relevant.b)In contrast, if the reference samples replicate a specific
task (e.g.,Indirect Object Identification), attribution scoresRlocalize the subgraph responsible for
that behavior, enabling circuit discovery.c)When the specific task is undesired (e.g.,toxic responses),
attribution scores highlight the circuit underlying the unwanted behavior. By contrasting relevance
scores seperatley drawn from general and toxic samples, we isolate and prune only the harmful
pathways; thus removing undesirable behavior while preserving the model’s general capabilities.

To address the efficiency challenges of LLMs, two widely studied approaches are pruning and
quantization. Pruning removes parameters that contribute little to the model’s predictions, thereby
increasing sparsity and reducing memory and compute demands. Successful quantization approaches
reduce the precision of model weights (e.g., from 32-bit float to 8-bitint), lowering the storage and
computational footprint without significantly affecting performance. Early foundational works in
pruning [ 17 , 14 ] propose using gradients to identify and eliminate irrelevant parameters. Recent
pruning techniques tailored to LLMs [ 38 , 16 , 18 , 31 , 39 , 18 ] focus on structural sparsity, per-layer
attribution scoring, and low-rank approximations to reduce model size while maintaining performance.
In this paper, we focus on pruning, specifically targeting parameters that are irrelevant to the model’s
inference process.

Understanding the internal mechanisms of Deep Neural Networks (DNNs) is a central goal of the
fields of eXplainable Artificial Intelligence (XAI) and mechanistic interpretability. Among the most
widely used tools in this area are attribution methods [ 4 , 30 , 32 , 21 ], which provide importance scores
for inputs or latent components, enabling the identification and interpretation of input features and
internal pathways most relevant to a model’s predictions [ 6 ]. Recent works have begun to explore
the utility of attribution methods for model compression instead. The works of [37, 40, 15] propose
using Layer-wise Relevance Propagation (LRP) [ 4 , 21 ] for structured pruning, with [ 37 ] focusing
on attention heads in language Transformers, and [ 40 , 15 ] targeting vision models. Notably, [ 15 ]
incorporates AttnLRP [1], an LRP extension for more faithful attribution of Transformer models.

A crucial step in attribution-based pruning [ 40 , 15 ] is the selection of reference samples – the input
examples used to estimate the importance of model components with – that strongly influence
which parameters are identified as relevant and, consequently, are retained or pruned. Using a
diverse set of general-purpose samples guides the pruning of parameters that contribute minimally
across tasks, enabling effective model compression. However, by selecting task-specific reference
samples, we can identify task-relevant subgraphs – also named circuits – which reflect the internal
pathways responsible for specific behaviors. This capability is of particular interest in mechanistic
interpretability [ 8 , 11 , 19 ]. Moreover, this approach enables targeted model editing. By using
reference samples that elicit undesired behaviors (e.g., the generation of toxic outputs), we can
attribute relevance to responsible components and selectively prune them in a post-hoc manner.


In this work, we propose a unified framework for attribution-guided pruning of LLMs supporting
three key applications as illustrated in Fig. 1: (1)General model compressionvia unstructured
pruning, (2)Circuit discoveryby extracting parameter-level subgraphs responsible for specific
tasks (e.g., indirect object identification); and (3)Model correctionby identifying and removing
circuits associated with undesired behaviors, enabling post-hoc editing with minimal impact on
overall performance.

## 2 Related works

Model compression and pruning The large size of LLMs leads to high memory and computational
demands. Compression mitigates these issues through techniques such as quantization, which lowers
parameter precision [ 7 , 16 , 39 ], and pruning, which removes parameters that contribute little to model
performance. Pruning strategies include knowledge distillation [ 22 ] or training with low-rank and
structured sparsity constraints [ 38 ], though these often incur high computational costs. Some methods
aim to prune with minimal fine-tuning [ 18 ], while others, such as [ 31 ], achieve efficient unstructured
pruning by identifying low-activation components only using forward-pass statistics on reference
samples. Unstructured pruning typically achieves higher sparsity levels, rendering it more effective
at reducing model size compared to structured and semi-structured approaches, but are less aligned
with current hardware accelerators [ 43 , 31 ]. In this work, we adopt an unstructured pruning approach
inspired by [ 31 ], but replace its activation-based heuristics with LRP [ 4 , 21 , 1 ], an attribution method
that has shown promise in the structured pruning of vision models [40, 15].

Circuit discovery Understanding LLM behavior is critical for improving safety and reliability,
especially in high-stakes applications. Circuit discovery, a central task in mechanistic interpretability,
aims to uncover the internal components, such as attention heads and Multilayer Perceptron (MLP)
neurons, that drive specific model predictions. Accurately extracting these circuits, however, remains
a challenge. Prior methods include Sparse Auto Encoders (SAEs) [ 19 ], which require training, and
activation patching (Automated Circuit DisCovery (ACDC)) [ 8 ], which ablates edges of the computa-
tional graph to assess importance but is resource-intensive and threshold-dependent. Alternatives
such as Information Flow Routes (IFR) [ 11 ] and Edge Attribution Patching (EAP) [ 33 ], streamline
the process (e.g.,by using gradients), but still rely on heuristics or external metrics. We instead
propose using LRP for efficient and scalable circuit discovery. LRP assigns relevance scores to
model components in a single forward-backward pass, enabling direct extraction of task-relevant
subgraphs. By ranking and pruning low-relevance components, LRP supports both structured pruning
(ofe.g.,attention heads, MLP neurons) and unstructured pruning (e.g.,individual weights). Unlike
token-level methods, our approach operates at the parameter level, naturally aligning with model
compression and behavioral control goals.

Model correction DNNs trained on large, imperfect datasets often exhibit undesirable behaviors,
such as shortcut learning, biased predictions, or toxic outputs. While data cleaning or fine-tuning
can mitigate these issues, such solutions are typically expensive and impractical at scale. Existing
methods address this in various ways. To mitigate this in vision models, the authors of [ 27 ] fine-tune
networks using a modified loss that leverages attribution information, while [ 29 , 3 , 24 , 10 ] identify
and remove biases by targeting directions in latent spaces. For LLMs, [ 26 , 35 ] edit model behaviors
exploiting specific directions in latent space, but these methods neither offer compression benefits
nor avoid fine-tuning. The authors of [ 23 ] align models with user intent via extensive fine-tuning,
while [ 9 ] localize knowledge neurons using gradients for behavioral control. In this work, we propose
a more efficient approach using LRP relevance scores to localize the components responsible for
undesirable behaviors. By comparing relevance from harmful versus benign reference samples,
we isolate and prune the responsible parameters. This yields targeted behavior correction without
fine-tuning, preserving performance while reducing model size.

## 3 Methods

We present a general framework for pruning deep models using attribution-based relevance scores.
We then introduce Layer-wise Relevance Propagation (LRP), the primary attribution method used in
our work. Finally, we define task-specific circuits and describe how their removal enables targeted
model correction.


3.1 Attribution-based pruning

Building on the framework introduced by [ 40 , 15 ], letΨ ={ψ 1 ,...,ψp}denote a set ofpcompo-
nents (neurons from MLPs, attention heads, or other trainable parameters) that constitute a DNN,
and letXref={x 1 ,x 2 ,...,xnref}represent a set of reference samples. For each componentψk∈Ψ
and reference samplexi∈Xref, we defineRψk(xi)as the relevance (or importance) score obtained
from an attribution method (i.e.,LRP). By aggregating these scores across all reference samples and
applying the normalization described in Eq. (1), we obtainR={R ̄ψ 1 ,R ̄ψ 2 ,...,R ̄ψp}, the set of
normalized relevance scores for all components.

```
R ̄ψk=^1
nref
```
```
Xnref
```
```
i=
```
```
Rψk(xi). (1)
```
Regardless of the pruning approach, whether it is structured, fully unstructured, per-layer unstructured,
or row-wise unstructured (an overview of these approaches is explained in Appendix C), we can
order the defined components based on their attributed relevance scores to receive the indicesc
corresponding to the least relevant components up to theq-th place:

```
{c}q=argsort(R) 1 , 2 ,...,q (2)
```
Defining 1 to represent an indicator function with conditioni∈{c}q, theqleast relevant components
can be pruned by masking as:

∀ψi∈Ψ :ψi7→(1− (^1) i∈{c}q)ψi (3)
3.2 Layer-wise Relevance Propagation
Layer-wise Relevance Propagation [ 4 , 21 ] treats a neural network withLlayers as a Directed Acyclic
Graph (DAG), such that for a given inputx:
f(x) =fL◦···◦fl◦fl−^1 ◦···◦f^1 (x) (4)
LRP employs a backpropagation process via specific rules designed to allocate “relevance” scores
to (both parametric and non-parametric) edges of the DAG, proportional to their contribution to the
final prediction. At first, this process begins at the last layerfLby initializing the relevance score of
RLjat outputjoffLand ultimately redistributing this score to its input variables. To elaborate the
redistribution at a specific layer ofl, denotezijto be the mappings of inputsito outputsjwhich in
linear layers this notation is represented byzij=aiwijwithwijas the weight parameters andaias
the activation of neuroni. LRP then redistributes the upper layer relevance quantity ofRljtowards
the lower layers proportionally to the relative contributions ofzijtozj, resulting inR(i←l−j^1 ,l)that
quantifies the contribution of neuroniat layerl− 1 , to the activation of neuronjat layerl:
R(il←−j^1 ,l)=
zij
zj
Rlj. (5)
An aggregation of allRi(←l−j^1 ,l)obtains the contribution of neuronito all upper layer neuronsj:
X
i
Rli−^1 =

### X

```
i,j
```
```
R(il←−j^1 ,l)=
```
### X

```
j
```
```
Rlj (6)
```
Extra steps on obtaining relevance scores from attention heads, and scores of each weight parameter,
are discussed in detail at Appendix B.1.


3.3 Circuit discovery

We define a circuit as a subnetwork comprising a subset of model componentsC ⊆Ψ, whereΨ
denotes the complete set of components (e.g., weights, neurons, or attention heads). A circuit is
extracted by iteratively pruning componentsψi∈Ψthat contribute least to a specific behavior, as
determined by their attribution scores computed on a set of reference samplesXrefdesigned to capture
the behavior of interest. During pruning, we ensure that the task-specific performance metric remains
above a predefined threshold. The resulting subsetCrepresents the essential components responsible
for the target behavior under sparsification.

In contrast, existing methods (e.g.,[ 8 , 33 , 11 ]) typically define circuits as computational subgraphs
derived from hidden activations across tokens, capturing information flow through the model for
specific inputs and producing circuits tied to individual examples. While these approaches reveal
detailed behavior for a given input, it makes the circuits hard to generalize and interpret. Our approach
instead identifies circuits directly from the model’s parameters by removing components that are not
important for a specific behavior. This yields input-independent circuits that are easier to interpret,
and more pactical for tasks like compression, analysis, and correcting unwanted behaviors.

3.4 Model correction

LetXrefGeneralandXrefUndesireddenote the sets of reference samples that respectively capture the model’s
general behavior (e.g.,Wikipedia and C4) and a specific undesired behavior (e.g.,toxicity). Applying
the framework described in Sec. 3.1 to each of these sets, yields two sets of attribution scoresRGeneral
andRUndesired. We then define a differential attribution setRdiff={R ̄diffψ 1 ,R ̄diffψ 2 ,...,R ̄diffψp}as:

```
R ̄diffψ
k=
R ̄Generalψ
k −
R ̄Undesiredψ
k (7)
```
Following the pruning procedure from Eq. (2), we sortRdiffin ascending order to prioritize the
removal of components the most responsible for the undesired behavior while being the least important
for the model’s general performance. This method resembles isolating and removing the part of the
undesired circuit that minimally overlaps with the subgraph governing the model’s general behavior.

## 4 Experiments

Our experiments cover the application and evaluation of our framework across the tasks of model
compression, circuit discovery, and model correction (see Fig. 1 for an overview over the tasks).

4.1 Unstructured pruning for model compression

We begin with model compression that has the aim to reduce model size without hurting model
performance on a general task. For model compression,unstructed pruningis the most widely used
approach due to its finer granularity and strong potential to achieve high sparsity with minimal impact
on performance. Compared to pruning individual components (e.g.,neurons or attention heads), it
allows selective removal of individual weights. As detailed in Appendix C, unstructured pruning can
be applied in various ways (i.e.,row-wise, layer-wise, or global).

Experimental settings We follow the evaluation protocol of [ 31 ], applyingrow-wise unstructured
pruning with uniform sparsity across the rows of weight matrices within linear layers. Thereby,
attribution scores are ranked by magnitude per row, rather than across layers or the full model, as
prior work [ 31 ] found global or per-layer ranking to yield inferior performance. To benchmark
our method, we compare against the state-of-the-art Wanda approach [ 31 ], which, like LRP, uses
reference samples to assign importance scores to parameters without relying on external metrics or
thresholds (see Appendix B.2). All experiments are conducted without fine-tuning. We evaluate three
models from the Llama family: TinyLlama [ 41 ], Llama-2-7B [ 34 ], and Llama-3-8B [ 2 ]. Performance
is assessed using two standard metrics: (1) perplexity on WikiText2 [ 20 ], reflecting uncertainty
in language modeling, and (2) zero-shot accuracy on a broad suite of tasks from [ 12 ], capturing
task-specific capabilities. Following [ 31 ], we perform attribution using reference samples from the
C4 dataset [ 25 ] to capture general model behavior. Specifically, we generate three sets of 128 samples
(sequence length 2048), each from a different random seed to ensure robustness.


Table 1: Perplexity (PPL) on WikiText2 and mean zero-shot accuracy (ACC) of TinyLlama, Llama2-
7B, and Llama3-8B under 50% sparsity via row-wise unstructured pruning. Errors represent the
standard error of the mean. Full performance details for each task are in the Appendix at Tab. 2.

```
Models
Method TinyLlama Llama2-7B Llama3-8B
(↓) PPL. (↑) ACC. (↓) PPL. (↑) ACC. (↓) PPL. (↑) ACC.
Original Scores 7.98 48.74 5.48 59.67 6.13 63.
Magnitude 24.42 41.38 16.13 51.26 206.82 38.
Wanda [31] 11.52± 0. 01 45.46± 0. 09 6.94± 0. 01 55.92± 0. 05 9.86± 0. 02 57.07± 0. 11
LRP 11.85± 0. 01 44.71± 0. 07 7.14± 0. 03 54.72± 0. 13 9.82± 0. 05 55.18± 0. 20
```
In Tab. 1, we apply a 50% sparsity rate. Higher sparsity rates (e.g.,60%) typically degrade model
performance strongly(as shown at Fig. 8 in the Appendix). Weight magnitude, a computationally
cheap compression method, is not effective in pruning, as larger weights do not necessarily indicate
greater contributions to decision-making – for example, a neuron with large weights may remain
inactive. Both LRP and Wanda perform well in unstructured pruning and model compression, with
Wanda showing a slight advantage. Our analysis in Appendix D.2 details the key methodological
differences between the two: Wanda efficiently attributes importance with fewer reference samples,
while LRP excels at identifying sparser, task-relevant subgraphs. Notably, LRP becomes more
effective when a larger corpus is available for attribution, which enables surpassing Wanda in
performance (also see Fig. 8 in the Appendix).

4.2 Discovering task-specific and sparse circuits

Understanding how specific behaviors’ functionalities are implmented within a model requires the
identification of sparse subgraphs – so-called circuits – that are necessary and sufficient for a given
task. In this experiment, we evaluate our framework’s ability to extract such circuits, focusing on
the well-established Indirect Object Identification (IOI) task [ 8 ], where the model must resolve for
the correct indirect object in a sentence. This task is frequently used to benchmark circuit discovery
methods due to its well-defined structure and known localization in models. Our goal is to assess
whether attribution-based pruning can recover circuits that preserve task behavior while achieving
high sparsity – i.e., pruning irrelevant components without affecting performance.

Experimental settings We use the 125M-parameter OPT model [ 42 ] and generate six reference sets
of 128 IOI-like sequences, following the data generation setup from [ 8 ], each sampled with a different
random seed. To extract circuits, we compare LRP and Wanda-based pruning and additionally
include gradient and neuron activation as baselines, following their use in [ 11 , 8 ]. All methods are
evaluated under two levels of granularity: (1)structuredpruning, where entire neurons or attention
heads are removed, and (2)unstructured pruning, where individual weight elements – edges between
neurnos – are pruned based on their attributed relevance.

A circuit is considered high-quality if it (i) includes all task-critical components (whose removal
significantly degrades performance) and (ii) excludes irrelevant ones. We assess this via performance-
sparsity curves, measuring task accuracy across a range of pruning rates. Inspired by the feature
perturbation paradigm for attribution evaluation [ 28 ], these curves reveal how resilient a circuit is to
pruning: a flat or even increasing trend suggests redundancy, while sharp performance drops identify
the pruning of essential components.

As shown in Fig. 2, relevance scores from LRP and Wanda produce significantly sparser parameter-
level IOI circuits compared to gradient and neuron activations. Results here align with [ 15 ], which
shows thatIntegrated Gradients(which is based on averaging gradients) [ 32 ] struggle with attributing
latent components due to noisy signals – an issue affecting gradients in general [ 5 ]. Further results in
Appendix E.1 and Appendix E.2 indicate that Wanda excels in row-wise unstructured pruning, while
LRP and gradient achieve superior results with globally unstructured pruning. However, under their
optimal settings (at Fig. 2), LRP consistently discovers sparser circuits, supporting our analysis in
Appendix D.2 where LRP is shown to better isolate task-relevant subgraphs. Moreover, Wanda is


Figure 2:a)IOI circuits are identified at the edge level – weight elements – within the linear layers of
the OPT model, specifically in the up and down projection layers of the MLP blocks (fc1andfc2).
For Wanda, row-wise unstructured pruning is applied. In contrast, for LRP and gradient, we perform
global sorting of components across all layers rather than within each row.b)IOI circuits extracted
within neurons of MLPs or attention heads via structured pruning generally exhibit lower sparsity
compared to unstructured pruning. The shaded region indicates the mean±standard deviation.

inherently limited in attributing components that involve multiple weights (e.g.,individual attention
heads) (see Appendix B.2). This limitation arises from Wanda’s implementation, where the attribution
process relies on assessing weight values and activations directly.

4.3 Model correction by supressing harmful circuits

This section addresses partcof Fig. 1 by combining circuit discovery and model compression to
suppress harmful behaviors in the OPT model. Controlling model behavior is crucial for ensuring
safety and trustworthiness, especially in sensitive applications where models may generate toxic,
biased, or harmful content. Undesired behaviors in LLMs extend beyond toxic outputs and can
include repetitive text generation, where models produce the same token or short sequences repeatedly.
Such repetitions degrade response quality and user experience, making their mitigation critical.

Experimental settings We here focus on toxic behavior and repetitive text generation. For toxic
behavior, we use the RealToxicityPrompts dataset [ 13 ], which provides prompts known to trigger
toxic responses, including profanity, gender bias, racism, and other harmful content. To quantify the
level of toxicity, we use the Perspective API^1 , which assigns a scalar values∈[0,1]to each model
response (higher scores indicate greater toxicity). We constructXrefToxicusing 93 prompts that generate
highly toxic responses (s≥ 0. 9 ). For text repitition, we construct a set of 53 prompts that consistently

trigger repetition measured by the Response Uniqueness Ratio (RUR) (r≤ 0. 5 ), formingXrefRepetitive

(see Appendix F.2 for more details). ForXrefGeneral, we use 128 randomly selected prompts from the
C4 dataset, similar to those in Sec. 4.1. We hypothesize that a subset of model components (a circuit)
is responsible for the individual undesired behaviors. Our objective is to identify and prune these
components, ensuring they are relevant to specific behavior (i.e.,viaR ̄ToxicorR ̄Repetitive) but have
minimal relevance to general tasks (viaR ̄General). This avoids degrading overall model performance.
Similar to Sec. 4.2, we compare LRP and Wanda and gradient for behavior suppression at multiple
levels of granularity: structured pruning (e.g.,removing neurons) and unstructured pruning (e.g.,
removing individual weight elements or edges between neurons).

Our model improvement results, shown in Fig. 3, reveal that removing just 100 (≈%0. 3 of total)
neurons from thefc1layers by using LRP in particular, significantly lowers the toxicity level

(^1) Jigsaw and Google. (2017).Perspective API. Retrieved fromhttps://perspectiveapi.com/


Figure 3:a)Pruning neurons from thefc1layers of the OPT model using attribution information
significantly reduces the toxicity measure. This has been achieved in its best case without affecting
the general performance (measured by perplexity on WikiText2). The shaded regions indicate the
standard error of the mean.b)The scatter plot illustrates per-sample toxicity changes in model
responses to prompts fromXrefToxicafter pruning 100 (≈%0. 3 of total) neurons infc1using LRP.
Detoxification effects vary across samples, but the average toxicity decreases.c)Example responses
after detoxification qualitatively demonstrate the method’s effectiveness.

Figure 4:a)Pruning approximately 7,000 (≈%0. 03 of total) weight elements from thefc1layers of
the OPT model by using LRP in particular reduces repetitive responses, measured using the Response
Uniqueness Ratio (RUR). This approach minimizes performance loss (perplexity on WikiText2). The
shaded regions indicate the standard error of the mean.b)The scatter plot shows per-sample RUR

changes to prompts fromXrefRepetitiveafter pruning with LRP. Effects vary, but the average uniqueness
increases.c)Example responses after pruning demonstrate reduced repetition.

of harmful responses. Notably, this detoxification is achieved without degrading general model
performance, as measured by perplexity on WikiText2. Extra results on other MLP layers and at
various pruning granularities, detailed in Fig. 13 and Fig. 14 in the Appendix, consistently confirm
the ability of our method to localize and prune toxic components without performance loss. Results
shown in Fig. 4 (with further examples at Fig. 15 and Fig. 16 in the Appendix) illustrate effective
suprresion of repetitive text generation without compromising general model performance. We focus
on moderate sparsity rates, based on the hypothesis that undesired behaviors are encoded in a small
subset of parameters. Higher sparsity rates caused significant performance drops, while very low
rates yielded minimal behavioral changes, indicating insufficient pruning. This supports targeting a
balanced range where harmful behavior can be mitigated without compromising overall performance.

Across both behavior correction tasks, the qualitative effects of pruning with different attribution
methods are illustrated in Fig. 5. While Wanda and gradient offer partial improvements and help
maintain model performance in certain configurations, LRP enables more reliable identification and


Figure 5: Model responses here qualitatively illustrate the effects of pruning-based targeted correction
using various attribution methods, including gradient, Wanda, and LRP. Panelsaandbrespectively
showcase the mitigation of toxic and repetitive responses. For toxic behaviors, we pruned 100
neurons, while for repetitive responses, approximately 7,000 weight elements were removed from the
fc1layers. Among the methods, LRP consistently demonstrates superior effectiveness in accurately
localizing and supressing undesired behaviors.

mitigation of harmful behaviors, demonstrating the generalizability of our method when a proper
attribution method is incorporated. Unlike fine-tuning, which is computationally intensive and risks
altering general model capabilities, our approach to pruning directly removes harmful parameters
while preserving general model behaviour, making it a lightweight yet effective solution.

## 5 Conclusion

In this work, we introduce a unified attribution-based pruning framework for three key applications
in Large Language Models: (1) model compression, (2) circuit discovery, and (3) targeted model
correction. Our method leverages attribution scores to identify parameters relevant to specific
behaviors, enabling fine-grained interventions without any fine-tuning. For compression, we show
that simple forward-pass-based attributions (e.g.,Wanda) are highly effective at identifying globally
unimportant weights. For nuanced tasks like circuit discovery and model correction, Layer-wise
Relevance Propagation proves more suitable, as it explicitly explains model outputs, thus identifying
task-specific components. By pruning parameters based on attribution scores, we recover sparse
subgraphs of a model (i.e.,circuits) that enable targeted correction of undesired behaviors by isolating
their internal mechanisms while maintaining performance.

Broader impact Our results highlight the potential of attribution methods not just for interpretability
but also for model compression and correction. This approach provides an efficient, interpretable
alternative to fine-tuning, enabling researchers and practitioners to compress, analyze, or control
LLMs. However, these capabilities also pose risks: the same techniques used to suppress harmful
behavior could, if misused, be leveraged to amplify it. This dual-use nature highlights the need for
careful ethical oversight and responsible deployment.

Limitations For model compression, we adopt row-wise unstructured pruning, following the
effective setup from [ 31 ] (see Appendix C). However, this may not be the optimal strategy. Future
work should investigate alternative pruning schemes or hybrid approaches, guided by attribution
scores. Another open challenge lies in selecting the appropriate granularity for circuit discovery and
model correction. As demonstrated in Sec. 4.2 and Sec. 4.3, the effectiveness of structured versus
unstructured pruning varies by context. Moreover, the specific layer types targeted for correction
significantly affect outcomes, suggesting a need for deeper analysis into which components most


influence task-relevant or harmful behaviors. Finally, our approach relies on the quality of reference
sets used to compute relevance scores. Reliable behavior correction requires reference samples that
accurately isolate the behavior of interest without overlapping with general capabilities. Future
research should explore principled methods for creating such behavior-specific reference sets to
improve attribution quality and intervention precision.

## Acknowledgments

This work was supported by the Federal Ministry of Education and Research (BMBF) as grant
BIFOLD (01IS18025A, 01IS180371I); the European Union’s Horizon Europe research and innovation
programme (EU Horizon Europe) as grant ACHILLES (101189689); and the German Research
Foundation (DFG) as research unit DeSBi [KI-FOR 5363] (459422098).

## References

[1]Achtibat, R., Hatefi, S. M. V., Dreyer, M., Jain, A., Wiegand, T., Lapuschkin, S., and Samek, W.
(2024). AttnLRP: Attention-aware layer-wise relevance propagation for transformers. InProceed-
ings of the 41st International Conference on Machine Learning, volume 235 ofProceedings of
Machine Learning Research, pages 135–168. PMLR.

[2] AI@Meta (2024). Llama 3 model card.

[3]Anders, C. J., Weber, L., Neumann, D., Samek, W., Müller, K.-R., and Lapuschkin, S. (2022).
Finding and removing clever hans: Using explanation methods to debug and improve deep models.
Information Fusion, 77:261–295.

[4]Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., and Samek, W. (2015). On
pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation.
PloS one, 10(7):e0130140.

[5]Balduzzi, D., Frean, M., Leary, L., Lewis, J., Ma, K. W.-D., and McWilliams, B. (2017). The
shattered gradients problem: If resnets are the answer, then what is the question? InProceedings
of the 34th International Conference on Machine Learning, volume 70, pages 342–350. PMLR.

[6]Bastings, J. and Filippova, K. (2020). The elephant in the interpretability room: Why use attention
as explanation when we have saliency methods? InProceedings of the Third BlackboxNLP
Workshop on Analyzing and Interpreting Neural Networks for NLP, pages 149–155. Association
for Computational Linguistics.

[7]Becking, D., Dreyer, M., Samek, W., Müller, K., and Lapuschkin, S. (2022). ECQx:
Explainability-Driven Quantization for Low-Bit and Sparse DNNs. InxxAI - Beyond Explainable
AI, Lecture Notes in Computer Science (LNAI Vol. 13200), Springer International Publishing,
pages 271–296.

[8]Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., and Garriga-Alonso, A. (2023). To-
wards automated circuit discovery for mechanistic interpretability.Advances in Neural Information
Processing Systems, 36:16318–16352.

[9]Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., and Wei, F. (2021). Knowledge neurons in
pretrained transformers.arXiv preprint arXiv:2104.08696.

[10]Dreyer, M., Pahde, F., Anders, C. J., Samek, W., and Lapuschkin, S. (2024). From hope to
safety: Unlearning biases of deep models via gradient penalization in latent space. InProceedings
of the AAAI Conference on Artificial Intelligence, volume 38, pages 21046–21054.

[11]Ferrando, J. and Voita, E. (2024). Information flow routes: Automatically interpreting language
models at scale.arXiv preprint arXiv:2403.00824.

[12]Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu,
J., Le Noac’h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L.,
Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A.
(2024). A framework for few-shot language model evaluation.


[13]Gehman, S., Gururangan, S., Sap, M., Choi, Y., and Smith, N. A. (2020). Realtoxicityprompts:
Evaluating neural toxic degeneration in language models.arXiv preprint arXiv:2009.11462.

[14]Hassibi, B. and Stork, D. (1992). Second order derivatives for network pruning: Optimal brain
surgeon.Advances in neural information processing systems, 5.

[15]Hatefi, S. M. V., Dreyer, M., Achtibat, R., Wiegand, T., Samek, W., and Lapuschkin, S. (2024).
Pruning by explaining revisited: Optimizing attribution methods to prune cnns and transformers.
arXiv preprint arXiv:2408.12568.

[16]Kim, S., Hooper, C., Gholami, A., Dong, Z., Li, X., Shen, S., Mahoney, M. W., and Keutzer, K.
(2023). Squeezellm: Dense-and-sparse quantization.arXiv preprint arXiv:2306.07629.

[17]LeCun, Y., Denker, J., and Solla, S. (1989). Optimal brain damage. Advances in neural
information processing systems, 2.

[18]Ma, X., Fang, G., and Wang, X. (2023). Llm-pruner: On the structural pruning of large language
models.Advances in neural information processing systems, 36:21702–21720.

[19]Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., and Mueller, A. (2024). Sparse
feature circuits: Discovering and editing interpretable causal graphs in language models.arXiv
preprint arXiv:2403.19647.

[20]Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2016). Pointer sentinel mixture models.
arXiv preprint arXiv:1609.07843.

[21]Montavon, G., Binder, A., Lapuschkin, S., Samek, W., and Müller, K.-R. (2019).Layer-Wise
Relevance Propagation: An Overview, pages 193–209. Springer International Publishing, Cham.

[22]Muralidharan, S., Turuvekere Sreenivas, S., Joshi, R., Chochowski, M., Patwary, M., Shoeybi,
M., Catanzaro, B., Kautz, J., and Molchanov, P. (2024). Compact language models via pruning and
knowledge distillation.Advances in Neural Information Processing Systems, 37:41076–41102.

[23]Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal,
S., Slama, K., Ray, A., et al. (2022). Training language models to follow instructions with human
feedback.Advances in neural information processing systems, 35:27730–27744.

[24]Pahde, F., Dreyer, M., Samek, W., and Lapuschkin, S. (2023). Reveal to revise: An explainable
ai life cycle for iterative bias correction of deep models. InInternational Conference on Medical
Image Computing and Computer-Assisted Intervention, pages 596–606. Springer.

[25]Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and
Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer.
Journal of machine learning research, 21(140):1–67.

[26]Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., and Goldberg, Y. (2020). Null it out: Guarding
protected attributes by iterative nullspace projection.arXiv preprint arXiv:2004.07667.

[27]Ross, A. S., Hughes, M. C., and Doshi-Velez, F. (2017). Right for the right reasons: Training
differentiable models by constraining their explanations.arXiv preprint arXiv:1703.03717.

[28]Samek, W., Binder, A., Montavon, G., Lapuschkin, S., and Müller, K.-R. (2017). Evaluating the
visualization of what a deep neural network has learned.IEEE Transactions on Neural Networks
and Learning Systems, 28(11):2660–2673.

[29]Schramowski, P., Stammer, W., Teso, S., Brugger, A., Herbert, F., Shao, X., Luigs, H.-G.,
Mahlein, A.-K., and Kersting, K. (2020). Making deep neural networks right for the right scientific
reasons by interacting with their explanations.Nature Machine Intelligence, 2(8):476–486.

[30]Smilkov, D., Thorat, N., Kim, B., Viégas, F., and Wattenberg, M. (2017). Smoothgrad: removing
noise by adding noise.arXiv preprint arXiv:1706.03825.

[31]Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. (2023). A simple and effective pruning approach for
large language models.arXiv preprint arXiv:2306.11695.


[32]Sundararajan, M., Taly, A., and Yan, Q. (2017). Axiomatic attribution for deep networks.
InProceedings of the 34th International Conference on Machine Learning, volume 70, pages
3319–3328. PMLR.

[33]Syed, A., Rager, C., and Conmy, A. (2023). Attribution patching outperforms automated circuit
discovery.arXiv preprint arXiv:2310.10348.

[34]Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra,
S., Bhargava, P., Bhosale, S., et al. (2023). Llama 2: Open foundation and fine-tuned chat models.
arXiv preprint arXiv:2307.09288.

[35]Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., and MacDiarmid, M.
(2023). Steering language models with activation engineering.arXiv preprint arXiv:2308.10248.

[36]Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and
Polosukhin, I. (2017). Attention is all you need.Advances in Neural Information Processing
Systems, 30.

[37]Voita, E., Talbot, D., Moiseev, F., Sennrich, R., and Titov, I. (2019). Analyzing multi-head
self-attention: Specialized heads do the heavy lifting, the rest can be pruned. InProceedings of
the 57th Annual Meeting of the Association for Computational Linguistics, pages 5797–5808.
Association for Computational Linguistics.

[38]Wang, Z., Wohlwend, J., and Lei, T. (2019). Structured pruning of large language models.arXiv
preprint arXiv:1910.04732.

[39]Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. (2023). SmoothQuant: Accurate
and efficient post-training quantization for large language models. volume 202 ofProceedings of
Machine Learning Research. PMLR.

[40]Yeom, S.-K., Seegerer, P., Lapuschkin, S., Binder, A., Wiedemann, S., Müller, K.-R., and
Samek, W. (2021). Pruning by explaining: A novel criterion for deep neural network pruning.
Pattern Recognition, 115:107899.

[41] Zhang, P., Zeng, G., Wang, T., and Lu, W. (2024). Tinyllama: An open-source small language
model.arXiv preprint arXiv:2401.02385.

[42]Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X.,
Lin, X. V., et al. (2022a). Opt: Open pre-trained transformer language models.arXiv preprint
arXiv:2205.01068.

[43]Zhang, Y., Wang, G., Yang, T., Pang, T., He, Z., and Lv, J. (2022b). Compression of deep
neural networks: bridging the gap between conventional-based pruning and evolutionary approach.
Neural Computing and Applications, 34(19):16493–16514.


## A Transformer Models

A.1 Llama and OPT

In the official implementations of Llama and OPT, the attention mechanism relies on projection
matrices obtained through individual linear layers. These matrices are key targets for pruning, as
they account for a significant portion of the model’s computational cost and memory usage. Both
Llama and OPT share a similar MLP architecture, using two linear transformations for up and
down projections of latent representations. These are labeled asup_projanddown_projin Llama,
andfc1andfc2in OPT. A notable architectural distinction is Llama’s use of an additional gating
mechanism, where an extra linear layer (gate_proj) applies an element-wise SiLU-activated gate,
enhancing the model’s expressivity.

## B Methods

In this section, more details on LRP [4, 21] and Wanda [31] will be elaborated.

### B.1 LRP

B.1.1 From neuron level to parameter-level attribution

As described in Sec. 3.2, LRP calculatesRlj, representing the relevance of neuronjin layerlfor the
model’s decision-making. For neuron-level pruning, where componentsψkcorrespond directly to
neurons, the attribution scoresR={R ̄ψ 1 ,R ̄ψ 2 ,...,R ̄ψp}are directly derived from these neuron

relevance values (Rlj). However, for unstructured pruning, relevance must be assigned to individual
weight elements rather than entire neurons. This requires a more fine-grained approach. Following
[ 7 ] and as shown in Fig. 6, LRP can be extended to compute relevance scores at the parameter level,
ensuring that each weight element is evaluated for its direct contribution to model decisions.

LRP is typically implemented as a modified gradient method, where the gradient is scaled by an input
term. As described in [ 7 ] and detailed in Eq. (8), LRP offers the flexibility to define this input term as
either the activationaior the weight parameterwij. The remaining component then serves as the
modified gradient. For our pruning approach, we adopt the latter formulation, treating the weightwij

Figure 6: a) Shematic architecure of a decoder-based transfomer demonstrates the sequential com-
bination of linear layers which constitutes the MLPs and attention heads. These individual layers,
involve a weight matrix denoted byWwhich is a favorable target for pruning. b) The formula
expressed in Sec. 3.2 shows how LRP attributes each individual neurons inside linear layers, making
it well-suited for structured pruning. c) However, for unstructured pruning of DNNs, the work of [ 7 ]
proposed an extra step to attribute individual weight elements, based on the scores initially computed
at the neuron level.


as the input. This allows us to directly compute a relevance score for each weight,Rwij=Ri←j,
effectively measuring the importance of individual parameters at any layerl.

```
Ri←j=aiwij
|{z}
zij
```
```
Rj
zj
```
```
| {z }
explicit
```
```
=aiwij
|{z}
∂zj
∂ai
```
```
Rj
zj
```
```
| {z }
mod. grad.
```
```
=wij ai
|{z}
∂zj
∂wij
```
```
Rj
zj
```
```
| {z }
mod. grad.
```
### . (8)

B.1.2 LRP rules

Several LRP variants exist, including LRP-ε, LRP-αβ, LRP-z+, and LRP-γ[ 4 , 21 ], each designed
to enhance stability and reduce noise. We adopt LRP-εin this work due to its robustness against
numerical instability, particularly division by zero in Eq. (5). This variant stabilizes computations by
adding a small constantε(typically1e− 6 ) to the denominator, defined as:

```
Ri←j=
```
```
zij
zj+ε·sign(zj)
```
```
Rj (9)
```
Note that here, [4] defines sign( 0 ) = 1 to achieve the desired stabilizing effect.

We leverage AttnLRP [ 1 ], following [ 15 ], to decompose LRP across attention heads, capturing the
contributions of each head through their associated softmax activations. This fine-grained attribution
is essential for accurately identifying task-relevant circuits within the attention mechanism.

B.2 Wanda

Unlike LRP, which requires both forward and backward passes to compute relevance scores, Wanda
[ 31 ] achieves efficient attribution using only a forward pass. It combines weight magnitudes and
activations to derive attribution scores for a given weight matrixWat layerlwith input activationsX,
computingRlWas:

```
RlW=|W|·||X|| 2 (10)
```
RlWhas the same dimensions asW. Each individual element ofRlWcorresponds to a relevance score
for the associated weight parameterwijinW.

Due to Wanda’s design, relevance scores cannot be assigned at the granularity of individual attention
heads, limiting its ability to capture fine-grained contributions compared to LRP. This limitation
stems from Wanda’s implementation, which is based on assessing weight values and activations
directly, rather than isolating the contributions of specific components. As a result, Wanda faces
challenges in attributing components that involve multiple weights, restricting its effectiveness in
tasks such as discovering circuits among the attention heads.

## C Pruning approaches

Several approaches can be used to apply pruning. The primary decision lies in choosing the granularity
level, indicating whether to prune entire neurons or individual weight elements, and later the scale of
comparison, which determines how the pruning rate is applied. For compressing LLMs, we follow
[ 31 ] and apply a uniform pruning rate to rows of weight matrices across all linear layers using a
row-wise unstructured approach. This method is illustrated in Fig. 7, which also compares alternative
pruning strategies.

In contrast to compression, we have followed these approaches for circuit discovery:

- Globally Strctured: We compute an importance score for each neuron (i.e., each row in the
    weight matrix) of the linear layers, then rank neurons across the entire model. This allows
    for comparisons between neurons across different layers, such as comparing neuroniin
    layerlwith neuronjin layerl+ 1.


Figure 7: Applying an individual pruning rate to a linear layer with the weight matrixW, yet not
limited to, can be achieved through either structured or unstructured approaches, potentially at various
granularities targeting the entire matrix or its individual rows. For the purpose of compression, this
paper adopts row-wise unstrctured pruning. In this figure, the zero-norm indicates the number of
non-zero elements inWdepicted with pink color

- Row-Wise Unstrctured: Following [ 31 ] and our experiments in Sec. 4.1, we apply uniform
    sparsity rates to each row of the weight matrix in the linear layer. Weight elements are
    compared within each row, as illustrated in Fig. 7.
- Globally Unstctured: This approach compares individual weight elements across layers,
    allowing direct comparisons betweenwijin one layer andwklin another. To reduce the
    computational cost of global sorting (which hasnlog(n)complexity), we use partitioning.
    However, this approximation may cause slight deviations from the desired pruning rate.

## D Compression with unstructured pruning

As discussed in Appendix C for compression via unstructured pruning, we adopt the row-wise
unstructured pruning method with a uniform sparsity rate across each row of the weight matrices in
the linear layers, as proposed in [31].

D.1 Zero-shot accuracies

Complete details on the zero-shot accuracy tasks are available in Tab. 2, with a summarized version
in Tab. 1. While perplexity measures model uncertainty, evaluating zero-shot accuracy across various
tasks provides insight into the reasoning capabilities of the compressed model.

D.2 LRP vs Wanda: core differences

In this section, we investigate the core differences between LRP and Wanda using the TinyLlama
model. Attribution scores were computed with a single random seed, following the experimental
setup of the unstructured pruning experiments in Sec. 4.1. We conclude by summarizing the key
observations from these comparisons.

Both Wanda and LRP rely on a set of reference samples, denoted asXref, for attribution. Comparison
of pruning performance with varying sizes ofXrefsheds light on the behavior of each method. As
shown in Fig. 8, Wanda [ 31 ] requires fewer reference samples to balance sparsity and performance
(measured by perplexity on WikiText2). In contrast, LRP shows performance instability with small
Xrefsizes but improves progressively as the sample size increases. With a large set of 8192 samples,
LRP achieves a perplexity of 11.35, outperforming other methods, as reported in Tab. 1.

To gain deeper insights, we compared the distribution of attribution scores from LRP and Wanda
under two scenarios: 1 ) attribution using 128 reference samples, each with a sequence length of
2048 tokens, and 2 ) attribution scores based on 3 individual samples, each of 2048 tokens. As
shown in Fig. 9, Wanda’s attribution score histograms remain largely consistent, even when only a
single sample is used. This stability is consistent with our earlier observations in Fig. 8, highlighting
Wanda’s ability to maintain reliable attribution across different sample sizes.


Figure 8: We performed row-wise unstructured pruning on TinyLlama using Wanda and LRP, testing
with varying sizes of the reference sample setXref. Three differentXrefsets were generated with
variable random seeds. Due to GPU memory limitations in Wanda’s official implementation, we were
unable to use reference sets larger than 1024 samples. The results also show that applying sparsity
rates above 50% significantly degrade model performance. The shaded regions in the figure represent
the standard deviations across the different seeds, providing an indication of the variability in the
results.

Figure 9: Attribution scores from Wanda and LRP were compared across three representative
layers of the TinyLlama model: the layer with the highest average importance (Layer 21, MLP,
down_proj), the median layer (Layer 6, MLP, gate_proj), and the layer with the lowest average
importance (Layer 0, Attn, k_proj). The layers were ranked according to the importance scores
computed by LRP. The histograms reveal that Wanda produces consistently stable attribution score
distributions across these layers, while LRP exhibits higher variability. Similar trends were observed
in other layers, though due to space constraints, we omit those visualizations.


Next, we investigate the sparsity of high-magnitude attribution scores in the TinyLlama model. Using
128 reference samples consistent with the settings in Sec. 4.1, we collected attribution scores across
all linear layers. These scores were min-max normalized to a range ofs∈[0,1], and we counted
those exceeding a given threshold. As shown in Fig. 10, LRP tends to concentrate importance on
a smaller subset of weights, while Wanda distributes importance more broadly, generally favoring
weights with large magnitudes and activations.
Based on our experiments (Sec. 4.1, Sec. 4.2, Sec. 4.3, and Fig. 8), Wanda effectively identifies
subgraphs relevant to model behavior with relatively few reference samples. However, as shown in
Fig. 10, these subgraphs are less sparse than those discovered by LRP. In contrast, LRP excels in
discovering sparse subgraphs, as evidenced by its superior performance in the circuit discovery task
(Sec. 4.2) and the higher sparsity of its attribution scores (Fig. 10). This method is particularly effective
when a larger reference set is available, as indicated by the variability in attribution histograms (Fig. 9)
and its gradual improvement with more reference samples (Fig. 8). In summary, LRP is more suitable
for identifying sparse, task-relevant circuits when ample reference data is accessible. Conversely,
Wanda is preferable for efficient compression when reference samples are limited.

## E Circuit discovery

```
In this section, we present the extracted IOI circuits of the OPT model accross a broad set of sparsity
rates, at other granularity levels based on the description in Appendix C. The experimental settings
are similar to the configurations from Sec. 4.2.
```
```
E.1 Circuit discovery via structured pruning
```
```
Circuits in this scenario are derived from neurons identified via globally structured pruning (Ap-
pendix C). As shown in Fig. 11, LRP-extracted circuits within the MLP layers are notably sparser and
deliver superior performance compared to alternative methods. In this context, activation information
was added as an additional baseline to align with ACDC [ 8 ]. Interestingly, all methods perform
poorly when targeting attention heads, underlining the crucial role of all heads working together in
the IOI task within the OPT model.
```
```
Table 2: Zero-shot accuracy of TinyLlama, Llama2-7B, and Llama3-8B models compressed at a 50%
pruning rate across tasks from [ 12 ]. The pruning rate is applied uniformly to rows of weight matrices
in the linear layers using the row-wise unstructured approach described in Appendix C.
```
```
Tasks
```
```
BQ RTE HS WG ARC-e ARC-c OBQA Mean
```
```
TinyLlama
```
```
Original ACC. 61.07 57.03 46.55 60.22 61.53 29.60 25.20 48.
```
```
Magnitude 54.40 55.59 36.53 54.77 47.18 22.61 18.60 41.
Wanda[31] 63.63 59.65 39.32 56.82 51.57 25.10 22.10 45.
LRP 62.77 58.12 38.77 55.95 52.11 24.34 20.93 44.
```
```
Llama2-7B
```
```
Original ACC. 77.73 62.45 57.17 69.29 76.55 43.08 31.40 59.
```
```
Magnitude 63.02 57.40 49.05 63.53 64.14 34.72 27.00 51.
Wanda[31] 75.85 53.42 52.64 67.67 71.96 39.07 30.80 55.
LRP 75.33 55.11 50.38 66.24 71.00 36.66 28.33 54.
```
```
Llama3-8B
```
```
Original ACC. 81.22 67.87 60.07 73.55 80.09 50.17 34.40 63.
```
```
Magnitude 42.66 53.06 29.87 52.32 46.46 25.00 22.00 38.
Wanda[31] 78.13 59.08 51.14 70.56 71.04 40.32 29.20 57.
LRP 73.87 56.55 49.84 68.66 71.45 38.62 27.26 55.
```

Figure 10: Number of attribution (relevance) scores exceeding a fixed threshold after min-max
normalization of Wanda and LRP scores across all 154 linear layers of the TinyLlama model. This
analysis reveals that LRP assigns importance to a sparser subset of components, whereas Wanda
distributes importance more broadly.

E.2 Circuit discovery via unstructured pruning

In this scenario, circuits are discovered from the weight elements attributed through globally un-
structured and row-wise unstructured pruning, as outlined in Appendix C. Wanda achieves the best
performance when IOI circuits are extracted using a uniform sparsity rate applied to rows of weight
elements within each linear layer, as shown in Fig. 12. This suggests that Wanda is particularly
effective when pruning is applied uniformly at the row level. On the other hand, LRP and gradient-
based methods yield superior results when weight elements are compared across different layers,
as illustrated in Fig. 12, indicating that these methods benefit from global pruning strategies. The
optimal configurations for Wanda, LRP, and gradient-based circuit discovery are summarized in
panelaof Fig. 2.


Figure 11: Based on the IOI circuits extracted from the OPT model using structured pruning, LRP
identifies sparser and more effective circuits across neurons in MLPs and attention heads compared
Wanda and gradient. Notably, due to the absence of explicit weight parameters for individual attention
heads in standard Transformer architectures, Wanda cannot be applied for circuit discovery within
these heads (see Appendix B.2). The shaded regions in the results represent the mean of standard
deviations, reflecting the variability in circuit discovery outcomes.

## F Model correction

F.1 Toxicity improvement

As detailed in Sec. 4.3, we identified the components of the OPT model responsible for both toxic and
general behaviors using LRP, Wanda, and gradient across various granularity levels. By pruning the
parameters contributing to toxic behavior, we effectively mitigated these behaviors while maintaining
the model’s overall performance. The results from structured and unstructured pruning, presented in
Fig. 13 and Fig. 14, respectively, highlight the superior effectiveness of LRP in reducing toxicity.

F.2 Repitition improvement

F.2.1 Repetitive response in LLMs

As explained in Sec. 4.3, depending on the prompt and temperature setting of the LLMs, models may
generate repetitive responses. These repetitions can manifest either as single tokens being repeatedly
generated or as entire sequences of tokens repeating. For instance:

Prompt: Stop, stop, stop,
Respone: stop, stop, stop, ...

Prompt: Love is something that
Respone: is shared by all. Love is something that is shared by all.

Prompt: If I repeat myself, then I repeat
Respone: myself. I repeat myself.I repeat myself. I repeat myself.


Figure 12: An overview of IOI circuits discovered from the OPT model using different layer types
and unstructured pruning approaches (described in Appendix C) is shown in the figure. Panels a and
b correspond to the row-wise and globally unstructured pruning approaches, respectively. Panel c
represents the best configuration for each attribution method, where the row-wise approach is optimal
for Wanda and the global technique is preferred for gradient and LRP. The shaded area in the figure
indicates the mean of standard deviations.

Figure 13: Removing neurons from different linear layers within the MLPs blocks of the OPT
model using structured pruning guided by attribution methods, effectively reduces toxicity without
degrading general performance (measured by perplexity on WikiText2). Among these methods, LRP
demonstrates superior effectiveness in minimizing toxicity while preserving model accuracy. The
shaded region in the figure indicates the standard error of the mean.


Figure 14: Pruning few weight elements across various MLPs layers improves the toxicity score of
generated responses, with methods such as LRP showing notable effectiveness compared to Wanda
and gradient. Here Wanda leverages row-wise while LRP and gradient use global unstructured
pruning. The shaded region in the figure indicates the standard error of the mean.

Such repetitive responses can be deliberately induced using specific decoding settings in the model’s
generation function. Specifically, settingtemperature=0,top_k=0, anddo_sample=Falseen-
sures deterministic (greedy) decoding, which is prone to repetition:

```
Generating Repetitive Responses with Hugging Face Transformers in Python
```
```
from transformers import AutoTokenizer, AutoModelForCausalLM
```
```
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
```
```
inputs = tokenizer("Love is something that", return_tensors="pt")
output = model.generate(
**inputs,
max_new_tokens=50,
temperature=0.0,# Deterministic (greedy) decoding
top_k=0, # No sampling, always choosing the most likely token
do_sample=False,# Deterministic generation
pad_token_id=tokenizer.eos_token_id,
)
```
These settings force the model to always select the most probable token at each step, increasing the
likelihood of repetitive outputs, particularly with certain prompts.

F.2.2 Quantifying repetitions

LetT= [t 1 ,t 2 ,...,tn]represent the sequence of tokens generated by model. We define the set of
unique tokens from the response asU={ti|ti∈T}. The Response Uniqueness Ratioris then
calculated as:

```
r=
```
### (

```
|U|
|T| if|T|>^0
0 if|T|= 0
```
### (11)


F.2.3 Reference samples triggering repetitions

We asked ChatGPT to generate a set of prompts that lead to repetitive responses, which we then
selected 53 samples from, each characterized by a low Response Uniqueness Ratio (r < 0. 5 ). These
prompts, detailed in Tab. 3, constitute the setXrefUndesired, used for attribution. For comparison, a general

setXrefGeneralwas created following Sec. 4.3. Our correction method (Sec. 3.4) effectively mitigates
repetition, as shown in Fig. 15 and Fig. 16, using structured and unstructured pruning, respectively.
Notably, LRP achieves superior performance, reducing repetition while preserving model perplexity
on WikiText2. This improvement is attained with fewer pruned parameters than other methods. Given
the substantial gains, we recommend unstructured pruning for optimal repetition reduction.

```
Table 3: List of 53 prompts that trigger highly repetitive responses for
the 125M-parameter of the OPT model, each resulting in a Response
Uniqueness Ratio ofr < 0. 5.
```
```
“Happiness can be
found in”
```
```
“If I had one wish, it
would be”
```
```
“Love is something
that”
“There is no doubt
that”
```
```
“I wake up every day
because”
```
```
“Sometimes, I wonder
if”
“The secret to success
is”
```
```
“Fate is what we make
of”
```
```
“Once upon a time”
```
```
“There was a boy who” “In a magical forest,
there lived”
```
```
“Long ago in a distant
land”
“In the middle of
nowhere”
```
```
“The princess
whispered to”
```
```
“It was the beginning
of the end when”
“The wizard cast a
spell and”
```
```
“Suddenly, the ground
shook and”
```
```
“Keep saying it:”
```
```
“I told you to
repeat:”
```
```
“Repeat this:” “Again and again I
say”
“Copy this:” “Echo these words:” “Repeat. Repeat.
Repeat.”
“I’ll say it again:” “Say it one more
time:”
```
```
“Can you say this
again:”
“Repeat this forever:” “Things I like:
apples, bananas,”
```
```
“My top five choices
are”
“These are my
favorites:”
```
```
“Consider the
sequence:”
```
```
“The next on the list
is:”
“Well, well, well” “So, so, so” “Like, like, like”
“Okay, okay, okay” “Um, um, um” “Hmm, hmm, hmm”
“Ah, ah, ah” “Alright, alright,
alright”
```
```
“Really, really,
really”
“Fine, fine, fine” “Maybe, maybe, maybe” “Hey, hey, hey”
“No, no, no” “Stop, stop, stop” “Listen, listen,
listen”
“Now, now, now” “I am what I am” “You are who you are”
“If I repeat myself,
then I repeat”
```
```
“There is no end to
this”
```
```
“And then, and then,
and then”
“Talking about
talking”
```
```
“Explaining an
explanation”
```
## G Used resources

All experiments were conducted on NVIDIA A100 GPUs (40GB). Both LRP and Wanda are efficient,
with LRP requiring a forward and backward pass for attribution (≈5 seconds per reference sample
with 2048 tokens on TinyLlama), while Wanda only uses forward passes (≈1 second per sample). As
model size or reference set grows, their computation times increase.


Figure 15: Reducing repetition in generated text can be achieved by selectively pruning neurons
from linear layers, particularlyfc1andfc2. Among the tested methods, removing just 20 neurons
via structured pruning significantly enhances the uniqueness of responses without compromising
model performance. Notably, gradient offers moderate improvements, while Wanda shows limited
effectiveness in this context. Specifically, LRP demonstrates superior performance, effectively
reducing repetition with minimal pruning. The shaded area in the figure represents the standard error
of the mean.

Figure 16: Following the approach in Fig. 15, enhancing the uniqueness of generated tokens while
maintaining model performance (measured by perplexity on WikiText2) can be achieved by pruning
a minimal number of edges (approximately 7,000 weight elements). Among the tested methods, LRP
demonstrates notable effectiveness, achieving these improvements with minimal sparsity rates. Here
Wanda leverages row-wise while LRP and gradient use global unstructured pruning. The shaded area
in the figure represents the standard error of the mean.


Memory-wise, LRP is more demanding, using≈60GB VRAM for a single 2048-token sequence on
TinyLlama due to stored activations and relevance scores. Wanda is lighter at≈5GB VRAM but can
become memory-intensive with large reference sets, as per the official implementation. Both methods
scale in memory usage with longer sequences or larger reference sets.


