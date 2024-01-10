---
layout: post
title:  "MoE Process Groups Configurations"
date:   2024-01-01 00:20:00 +0000
categories: mixture-of-experts
usemathjax: true
---

In recent years, the evolution of AI has been marked by a immense shift towards the scaling of language models/LLMs, characterized by the exponential growth in data, parameters, and model sizes. This transformation has yielded remarkable advancements in the field of NLP, empowering the accomplishment of more complex tasks, enhanced reasoning abilities, and the utilization of fewer labeled data. Notable milestones are include the development of renowned models like *GPT-2, BERT, T5, GPT-3, FLAN, Gopher, Chinchilla, and PaLM.*

## Brief Introduction to MoE

While the current shape of language models continues to demonstrate effectiveness and high evaluation scores, the rising costs and energy consumption demands associated with further scaling are becoming overwhelmingly burdensome *(Carbon Emissions and Large Neural Network Training, Patterson et al.,2021)*.

In response to this consumption growth, considering alternative architectures, such notable thirty-year-old but emerging sparsely expert models become an active area of research and experimentation in the context of large-scale deep learning. Various studies that are focusing on sparse expert models, demonstrated their capacity to yield novel performance improvements through the exploitation of neural network sparsity. This is achieved in conjunction with memory-efficient training and inference techniques, as an alternative to the conventional densely connected models. Ultimately, these approaches have shown their promise in reducing computational and energy requirements.

These sparse expert models include architectures such as *Mixture-of-Experts, Switch Transformers, Routing Networks, Hash layers, and BASE layers*. The common thread among these models is the shared idea that each individual instance is influenced by a specific subset of the parameters. 

Many modern sparse expert models drew inspiration from the introduction of a new type of general-purpose neural network component: a Sparsely-Gated Mixture-of-Experts Layer (MoE) which comprises several experts, each functioning as a simple feed-forward neural network. This concept originated from the research titled *'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer'* authored by Shazeer et al. in 2017 as part of Hinton's Group at Google. Before all, for twenty years of MoEs, it may be useful to look at this comprehensive survey: *Seniha Esen Yuksel, Joseph N Wilson, and Paul D Gader. Twenty years of mixture of experts. IEEE transactions on neural networks and learning systems, 23(8):1177–1193, 2012.*

The idea of a mixture-of-experts was already established three decades earlier, building upon the work of *Jacobs et al. (1991)*, *Jordan and Jacobs (1994)*. In the early concepts, the experts constituted entire neural networks, and the MoE resembled ensemble methods more closely. However, it wasn't until ready the work of Shazeer et al. (2017) that the first large-scale success with this approach was achieved.

A promising alternative that enables the scaling of models in size without incurring their full computational cost is the use of sparse mixtures of experts. 

This concept of sparse experts and its variants, with the most common variant being sparse expert models, harnesses the advantages of neural network sparsity. It allows networks to allocate different subsets of model weights to their inputs, resulting in significantly larger models with smaller computational footprints.

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/moe-groups-configurations/9.png">
</figure>

This approach has gained popularity within the NLP domain and has proven to be particularly well-suited for handling larger models. By leveraging the abundance of billions and trillions of tokens, especially in the context of tasks like next-word prediction, masked language modeling, and even vision-related tasks.

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/moe-groups-configurations/8.png">
</figure>

*[Schematic Diagram of MoE Transformer](https://www.researchgate.net/figure/Schematic-diagram-of-MoE-Transformer-encoder-prominent-improvement-by-modifying-the-loss_fig1_357013990)*

Moreover, the popularity of sparse expert models saw a significant boost when integrated with the current nearly de facto standard, the transformer architecture. Within Transformer models, MoE layers are frequently employed to select the FFN layers, which appear in each Transformer block following the multi-headed attention mechanism. Despite earlier success, GShard effected the research of MoE + Transformer models. Latest systems gone further with improved training and deployment of MoE models. 

Below, comparison of dense - sparse model with transformer architecture can be seen:

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/moe-groups-configurations/1.png">
</figure>

*[A Review of Sparse Expert Models in Deep Learning](https://arxiv.org/abs/2209.01667)*

Models such [Switch Transformer](https://arxiv.org/abs/2101.03961), [GLaM](https://arxiv.org/abs/2112.06905), [V-MoE](https://arxiv.org/abs/2106.05974) have demonstrated better scaling in multiple domains and better retention capability in a continual learning setting with the help of employing a sparse and discrete router that tries to find a good hard assignment between tokens and experts inside the sparse MoE layers which is an extension of the MoE model to deep neural networks.

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/moe-groups-configurations/2.png">
</figure>

*Architecture from the Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.*

Now, there is widespread speculation surrounding GPT-4's use of Mixture of Experts, thought to incorporate 16 experts, each boasting approximately 111 billion parameters. 

Essentially, MoE layer is comprises a set of M "expert networks" and a "gating network", typically linear in nature. All these experts have identical network architecture and are trained with the same algorithm. These experts are controlled by a gating network or routing mechanism, which directs specific inputs towards a select few experts from the available layer. Using a sparse gating function, the MoE layer's router can route each input to the "top-K experts", where K is greater than or equal to 2, or alternatively to the singular top expert when K equals 1.

When an input is provided, it's not immediately passed to every expert. Instead, the gating network first processes this input to determine which experts should be activated for this specific input. The gating network outputs a set of weights for each expert, effectively scoring each expert based on its expected relevance to the current input. In most MoE implementations, this gating network uses a softmax linear unit function to ensure that the weights sum to one, thus providing a kind of probability distribution over the experts.

Depending on the design, the top-K experts *(as determined by the gating mechanism)* will process the input. If k=1, only the most relevant expert will process the input. Once the selected experts have processed the input, their outputs are combined. This is typically done using a weighted sum, where the weights come from the gating network. At the end of the day, this generalized setup *(as demonstrated in GShard)* allows the model to handle different parts of the input space with specialized experts and computes the union of all group-local of gating function for parameters E *(number of experts)* and D *(model embedding dimension)*.

After MoE layers have a place for performance in both large transformer models and deep learning, models and optimization libraries adopted MoE layer integrations within their specific wrappers. One of the libraries early adopted MoE integration was Microsoft's open-source DeepSpeed library.

With the release of DeepSpeed v0.5, DeepSpeed team introduced MoE layer API, an API designed to bring the training and inference of a class of sparse MoE layers characterized by computational costs scaling sublinearly in relation to their parameters. Supported with various forms of internal parallelism approaches for both training and inference processes, by effectively utilizing both GPU and CPU memory resources. Showcased numerous proof-of-concept demos, such as the CIFAR training with MoE, to provide insightful examples of the MoE API in action.

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/moe-groups-configurations/3.jpeg">
</figure>

*[DeepSpeed with Numbers](https://www.microsoft.com/en-us/research/blog/deepspeed-advancing-moe-inference-and-training-to-power-next-generation-ai-scale/)*

**Now, from this point of this post, I want to discuss the how process groups contributes for the large-scale training and inference of MoE layers in DeepSpeed.** As an entry point, library brings a capability known as *'expert group initialization'* enabling users to intricately combine different, sophisticated parallelism techniques. The combination of these adaptable parallelism configurations internally generates diverse groups of processes during the initialization of the MoE layer API. Library currently offers two type of MoE layer: "Standard" & "Residual" with this window. Each MoE layer has the flexibility to define its unique expert parallelism degree and vary the number of experts in accordance with introduced parameters, named 'ep_size' and 'num_experts'. Enabling mixture-of-expert models with a scale of hundres of billions and trillions of parameters necessitates a highly intricate amalgamation of multiple parallelism methods that did not exist prior to the launch of this process group initialization feature.

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/moe-groups-configurations/4.png">
</figure>

*High-level overview of MoE packages*

## Breaking DeepSpeed MoE Layer Into Smaller Units

Here is the class structure of the DeepSpeed MoE layer, broken down independently to make it more understandable for us with individual units.

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/moe-groups-configurations/5.png">
</figure>

*Class structure of DS MoE API*

This whole class architecture represents MoE layer computations with memory-efficient parallelization, with a strong emphasis on distributed processing and efficient handling of inputs, a high-performance setup designed for large-scale or complex tasks.

### MoE Entrypoint

Overarching MoE entrypoint class accepts many initialization parameters including hidden dimension, DeepSpeed MoE entry module that defines experts, top-k gating value*(only can be *k=1* and *k=2*; refers to selecting the top-k experts - based on the gating mechanism's scores)* for a given input *(- basically determines how many experts are engaged for a given input)*, capacity options, residual MoE initialization selection *(SoTA technique for MoE layer, developed by DeepSpeed Team)*, gating policy *(configuration of 'Jitter', 'RSample', or 'None', defining how the gating mechanism should behave)*, token dropping and expert tensor parallelism selection.

This overarching MoE class provides initialization parameters for the MoE computation layer. When instantiated, an object of the MoE class becomes an MoE layer that can be integrated into a neural network model.

Furthermore, if the user requires process groups, the class will create parallelism process groups. If tensor parallelism is not set, or if expert tensor parallelism is intentionally disabled by the user, the library will create a *simpler parallel group* as outlined in the DS documentation's E+D parallel section. Otherwise, it will create an E + M + D group.

After determining and possibly creating the appropriate group, as an interceptor before computation happens in the MoELayer, Experts class builds a list of experts, preparing each expert for distributed training scenarios, disables *all-reduce operations* for the parameter during distributed training, assigns a group name to each parameter, and eventually returns prepared experts for MoELayer set up and making them ready for further operations or for integration. Then, creating it sets this group for the main MoE layer. We will simply inspect the specifications of these groups in the following *Group Configurations* section.

### MoE Computations

After layer parameters are passed to the "MoELayer", which is implemented according to the definitions in the *[GShard paper](https://arxiv.org/pdf/2006.16668.pdf)*. When user pass parameters and other layer parameters to the MoELayer, essentially priming the layer for specialized processing for *Top-K-Gating mechanism computations* (algorithm 1 or algorithm 2). There are many kinds of Top-K-routing algorithms, however, DeepSpeed uses the routing mechanism proposed in Shazeer's work e.g. DSelect-k, which is a smooth version of the top-k routing algorithm that improves over standard top-k routing.

Top-K-Gating acts a dynamic router and selecting the top-k experts based on their scores. Forward pass inside Top-K-Gating class, takes in an input tensor and computes which experts should handle it. Logits for each expert are then computing using the linear layer *(wg)*.

Then top-k-gating mechanism, represent the suitability scores of experts for the input. Depending on the value of *k (1 or 2)*, *(setted in the MoE layer parameters)* it either uses the top-1 gating mechanism *(top1gating)* or the top-2 gating mechanism *(top2gating)* to compute the final gating output. Finally, returns the gating output.

Top-K-Routing mechanism here is achieved by adding sparsity and noise to standard softmax gating function. Before taking the softmax function, Top-K-Gating adds tunable Gaussian noise, then keep only the top k values, setting the rest to −∞. Tunable implies that the standard deviation of the Gaussian noise can be adjusted. Recall that we define:

\\(G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))\\)

GShard-based gating mechanisms in the gating function GATE(·) implements requirements for expert capacity, local group dispatching, auxiliary loss, intuitively random routing.

Once the Top-K-Gating mechanism has been prepared, orchestrator forward pass computations starts inside the MOELayer class, after computations returns the output tensor, l_aux tensor, number of expert counts to the MoE layer and aggregating the expert predictions. Depending on whether residual MoE is enabled, the output tensor may also be combined with the output of an MLP using a learned coefficient. But eventually, it returns the output tensor with the l_aux *(group auxilary term - the term \\(c_{e}/S\\) represents the fraction of input routed to each expert, and we want to minimize mean square of \\(c_{e}/S\\))*, *num(exp_counts)*.

For MoE gating mechanism, DeepSpeed uses sparse data structures instead of commonly used dense representations that contains cubic number of zeros and quadratic number of non-zeros with respect to the number of input tokens. This approach reduces the compute complexity from cubic to quadratic *(as explained in "Kernel Optimizations", in the paper "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale")*. Efficient implementations of the all2all primitive inside DeepSpeed, along efficient implementation of routing algorithms, reduces the added communication costs from sparse expert models.

When we construct an simple MoE model like this, below computations starts and returns the outputs of our MoE layer for our training and inferencing:

```python
class SimpleMoEModel(torch.nn.Module):
    def __init__(self, hidden_dim, num_experts=4, ep_size=1, use_residual=False):
        super(SimpleMoEModel, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        expert = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Linear(hidden_dim, hidden_dim))
        # using two MoE layers to check implications of sharing a single storage
        self.moe_1 = MoE(hidden_size=hidden_dim,
                         expert=expert,
                         ep_size=ep_size,
                         use_residual=use_residual,
                         num_experts=num_experts,
                         k=1)
        # interleaving MoE modules with dense to create an opportunity
        # for gradients to be merged in ZeRO stage 2 average_tensor reduce bucket
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.moe_2 = MoE(hidden_size=hidden_dim,
                         expert=expert,
                         ep_size=ep_size,
                         use_residual=use_residual,
                         num_experts=num_experts,
                         k=1)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = self.linear1(x)
        output, _, _ = self.moe_1(hidden_dim)
        output = self.linear2(output)
        output, _, _ = self.moe_2(output)
        output = self.linear3(output)
        hidden_dim = hidden_dim + output
        sentence_embed = hidden_dim.mean(1)
        return self.cross_entropy_loss(sentence_embed, y)
```

For inferencing we get the created MoE model and initializating DeepSpeed inferencing engine, for inferencing MoE models you can [check](https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/) the detailed documentation of the library.

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/moe-groups-configurations/6.png">
</figure>

*Prestaged inference performance results according to the documentation*

**P.S.** It is possible to create a config setup of MoE layer for inference, it can be accessible within *`deepspeed.inference.config.DeepSpeedMoEConfig`*. The library constructs a DSMoEModelConfig, which stacks a set of parameters for the MoE inference.

## Process Groups

DeepSpeed's advanced abstracted kernels  generate various process groups based on parameters such as the user's model parallelism degree *(mp_size)*, expert parallelism degree *(ep_size)*, and the number of experts (moe_experts). Runtime engine checks these parameters at the backend alongside of entire pipeline. Sets the group handle for the returned MoELayer object. Based on entry conditions, the function calls appropriate methods to create process groups.

Because of the DeepSpeed MoE training systems is specifically designed for the efficient scaling of MoE models. It supports models that are up to [8 times larger](https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf) by utilizing a versatile mix of parallelism techniques. Include these; tensor slicing, data parallelism, ZeRO-powered data parallelism, and expert parallelism. These process groups are created considering the smaller of the two values: total num of GPUs (known as *world_size*) and expert parallel size.

**Expert Parallelism (E):** This involves distributing different "experts" or parts of MoE layer across multiple devices. In a MoE layer, different experts specialize in different parts of the input space. Expert parallelism allows distributing these experts across multiple GPUs or other hardware accelerators. Doesn't perform all-reduce operations among themselves but use all-to-all communication patterns.

**Data Parallelism (D):** This is the traditional parallelism where a mini-batch of data is divided across multiple devices. Each device computes the forward and backward passes for its chunk of data. Manages data-level parallelism and is responsible for all-reduce operations but only on the MoE parameters. So, restricted to the MoE parameters.

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/moe-groups-configurations/7.png">
</figure>

*Currently stated groups are can be seen in the table from the official documentation*

- Condition I *(TP not enabled)* generates expert + data parallelism (E+D) group without tensor parallelism. Doesn't rely on an MPU (model parallel unit).

- Condition II *(TP enabled)* generates expert + data + model parallel groups based on MPU (model parallel) group. Specifically, designed for a more complex parallelism setting where parts of the model are also distributed across multiple devices.

According to below E + D and E + D + M example configuration would be:

```python
# For E + D parallelization
world_size = 16
expert_parallel_size = 2 # number of experts in same group
expert_data_parallel_group = [0,2,4,6,8,10,12,14], [1,3,5,7,9,11,13,15] - all reduce is only on MoE params
expert_parallel_group = [0, 1], [2,3], [4,5], [6,7], [8,9] - no all reduce, but all to all
data_parallel_group = [0,1,...,15] - all reduce is only on non-MoE
```

```python
# For E + M + D parallelization
world_size = 16
model_degree = 2
expert_degree = 4 # number of experts in same group
mp_group = [0, 1], [2,3], [4,5] ... # that defines tensor parallel
data_parallel_group =[0,2,4,6,8,10, 12,14],                 [1,3,5,7,9,11,13,15]
expert_parallel_group = [0,2,4,6], [8,10,12,14]             [1,3,5,7], [9,11,13,15]
expert_data_parallel_group = [0,8],[2,10],[4,12],[6,14],    [1,9],[3,11],[5,13],[7,15]
```

**If user initialize an E + D + M configuration:** DeepSpeed first collects the world size and ranks from the distributed system, checks if the world size is divisible by the model (guarantee for experts are evenly distributed across the available devices) and expert parallel sizes, calculates the ranks, populates `_EXPERT_PARALLEL_GROUP` and `_EXPERT_DATA_PARALLEL_GROUP` dictionaries with the newly created groups.

**If user initialize an E + D configuration:**  It also gets the 'world_size' and fetches the 'rank' of each device. Again, guarantees that the experts are evenly distributed across the available devices. Initializes groups function for a system where user are only concerned with E + D parallelism.

**Without model parallelism,** user avoid the communication overhead that comes from splitting the model across multiple GPUs. Because of that, those models that can fit into the memory of a single GPU, E+D should be sufficient. It is expected to be allow user to add more experts easily, providing a flexible way to scale the model.

**However,** when models are too large to fit into a single GPU device, model parallelism becomes necessary. E + D + M flexible group provides an optimized way to train large models that need to be both sharded across GPUs (MPU) and have data distributed across them (DP).

Additionally, when user creating a MoE configuration, if the runtime engine detects the Zero Redundancy Optimizer (ZeRO) *(a novel memory optimization technology in the library)* enablement, enables methods to removal of the memory redundancies across data-parallel processes by partitioning the three model states *(optimizer states, gradients, and parameters)* across data-parallel processes instead of replicating them. Applies ZeRO configuration alongside our parallelism groups, generates through ZeRO enablement, the system boosts memory efficiency compared to traditional data-parallelism, while maintaining its computational precision and communication effectiveness.

Currently, Zero Redundancy Optimizers has 3 different stages: 

- ZeRO Stage 1
- ZeRO Stage 2
- ZeRO Stage 3 *(addition with Infinite Offload Engine)*

Considering all this, DeepSpeed brings an amazing abstraction of Zero optimizers *(extremely complex kernels in their nature)* with ease for training expert models. To enable Zero Optimizer stages for MoE training it is required to enable - moe-param-group parameter. Because the optimizer can treat these groups differently based on their specific MoE configurations, it can also manage MoE parameters differently from non-MoE parameters. Then created, param groups can be fed to the ZeRO stage-2 optimizer as follows as shown in the documentation:

```python
net = Net()
parameters = create_moe_param_groups(net)
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=net, model_parameters=parameters, training_data=trainset)
```

After, passing into the initialization method of DeepSpeed engine, it is required to enable `zero_optimization`` config flags with stage seperations and customizable ZeRO operation parameters. For detailed instructions of the ZeRO enablement in the library, related documentation can be found in [here](https://www.deepspeed.ai/tutorials/mixture-of-experts/#combining-zero-offload-and-deepspeed-moe-for-very-large-models). In this way, based on the stage parameter configuration E + D + Z and E + Z-Off + M can be configured as articulated flexible parallelism group of DeepSpeed.

## References

- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale [arXiv:2201.05596](https://arxiv.org/abs/2201.05596) [cs.LG]
- Carbon Emissions and Large Neural Network Training [arXiv:2104.10350](https://arxiv.org/abs/2104.10350) [cs.LG]
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) [cs.LG]
- GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding [arXiv:2006.16668](https://arxiv.org/abs/2006.16668) [cs.CL]
- A Review of Sparse Expert Models in Deep Learning [arXiv:2209.01667](https://arxiv.org/abs/2209.01667) [cs.LG]
- Adaptive Mixture of Local Exports [https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf]()