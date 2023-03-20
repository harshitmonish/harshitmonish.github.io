---
title: 'What are Graph Compilers? (LLVM, MLIR)'
date: 2023-01-10
permalink: /posts/2012/08/blog-post-4/
tags:
  - Artificial Intelligence
  - Deep Learning
  - Compiler Design
  - Graph compilers
  - LLVM
  - MLIR
---
Graph compilers Introduction
===
<br/><img src='/images/graph_compiler_2.png'>

With the increase in popularity of Artificial Intelligence applications many machine learning and deep learning frameworks have been developed to create ML/DL models e.g. Tensorflow, PyTorch, Keras, mxNet.With dataflow at the heart of most of these computations, efforts have been made to improve performance through graph compilation techniques.
AI training is computationally intensive and High Performance Computing has been a key driver in AI growth. AI training deployments in HPC or cloud can be optimised with target-specific libraries, graph compilers, and by improving data movement or IO. 

Graph compilers aim to optimise the execution of a DNN graph by generating an optimised code for target hardware thus accelerating the training and deployment of DL models.
The Deep learning models are usually represented as computational graphs, with nodes representing tensor operators, and edges the data dependencies between them. This computational graph is then used to further optimise for different hardware back-ends which include operator fusion, memory latency hiding, etc.

With the increase in ML/DL frameworks and also increase in hardware optimised for different ML use cases, e.g. GPU, TPU, AI accelerators, question arises how do we make a model built with an arbitrary framework run on arbitrary hardware?  
<br/><img src='/images/graph_compiler_1.png'>

Providing support for a framework on a type of hardware is tedious, time-consuming and engineering intensive. A fundamental challenge is that different hardware types have different compute primitives and different memory layouts. Deploying ML models to new hardware such as mobile phone, FPGAs, embedded devices, GPUs, etc. requires mutual effort. some examples of graph compilers are:
* [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) : This compiler is built on top of CUDA and optimises the inference by providing high throughput and low latency for deep learning inference applications. It supports ONNX, thus by extension supporting models trained by different frameworks, it provides optimisations based on reduced precision and hence provide faster performance compared to CPU-only platforms during inference.
* [Intel nGraph](https://www.intel.com/content/www/us/en/artificial-intelligence/ngraph.html) : It is an end-to-end compiler for training and inference, it supports TensorFlow, MXNet, ONNX, etc. It can deliver increased normalized inference throughput leveraging MKL-DNN on Intel Xeon Scalable processor.
* [GLOW](https://arxiv.org/abs/1805.00907): It optimises Neural Networks by lowering the graph to two intermediate representations. Glow works with PyTorch and supports multiple operators and targets. Glow can consume ONNX (open standard for serializing AI model) as an input and thus can support other frameworks.
* [XLA](https://www.tensorflow.org/xla) : This compiler accelerates linear algebra computations in TensorFlow models and achieves a 1.15x speedup when enabled on standard benchmarks.

Intermediate Representation
===
For every new hardware type and device, instead of targeting a new compiler and libraries, we can create a generic representation which bridge frameworks and platforms. Framework developers will no longer have to support every type of hardware and only need to translate the framework code to Intermediate Representation and hardware engineers can support this IR after optimizations for respective hardware. IR lies at the core of how these compilers works, the orginal code of the model is converted into series of high and low level Intermediate Representations before generating hardware-native code to run your models on a certain platform.

High level IRs are generally hardware-agnostic(doesn't care what hardware they'll be running), while low-level IRs are framework-agnostic(doesn't care what framework the model was build with). To generate machine-native code from an IR, compilers typically leverage a code generator, the most popular codegen used by ML compilers is [LLVM](https://en.wikipedia.org/wiki/LLVM). XLA, NVIDIA CUDA Compiler (NVCC), TVM, MLIR all use LLVM.

LLVM 
===



Model Optimization techniques
====

With such huge models come the huge computational cost for inference and training these models in cloud infrastructure. AI model optimization becomes a key aspect to reduce the cost and can be performed in both training and inference. Following are the few optimization techniques:
* Quantization : quantization is where we reduce the number of bits used to represent our data and weights, e.g. reducing 32 bit number to 16 bit and 8 bit number. This reduction with bits when combined with an architecture can result in significant increase in performance or to reduction in power or both. Challenge is how to map these values from higher number of bits to lower number of bits and maintain an acceptable performance.
* Pruning is another optimization of deep learning models that reduces the size of the model by finding small weights and setting them to zero. Deep learning models typically have large number of weights very close to zero which have minimal contribution to model inference, hence can be ignored. There are several approached to prune a model:
  * Structured and Unstructured techniques: In unstructured technique we remove the weights on a case to case basis whereas in structured
approach we remove weights in groups e.g- removing entire channel at a time. Structured
pruning typically has better run-time performance characteristics but also has a heavier
impact on accuracy of the model
  * Scoring : This approach takes into consideration the threshold value to discard weights . this
threshold value is either computed using L1 pruning scores by computing their contribution
to the overall tensor vector using taxicab distance or using L2 pruning score computing
using Euclidean distance.
  * Scheduling : : In this approach we add additional steps in the training process to prune the
model after few epochs. This is also applied in between fine-tuning steps.
* Knowledge Distillation: Knowledge distillation is another technique of model optimization in which we transfer the knowledge from a large to a smaller model that can be practically deployed under real-world constraints. A small student model learns to mimic a large teacher model and leverage the knowledge of the teacher to obtain similar accuracy.

OneDNN, CuDNN are some of the libraries that are highly optimized for linear algebra operations that executes fast on the targeted hardware backend.

AI/ML Model Instruction Analysis
======

If we closely analyse the Neural networks operations majority of them are general matrix multiplication computations. These can be decomposed into several multiply and addition or MAC operations which can execute in parallel. Hence, hardware supporting high number of MAC operations execution in parallel (e.g. GPU) can be advantageous in training process.

During backpropagation stage each training iteration, we need to use the output from each layer that were computed in the forward propagation stage. These values are typically stored in memory after being computed during the forward propagation and read back and used during back propagation. Thus, a training iteration involves moving large amount of data from compute to memory and vice versa. Therefore, hardware supporting high bandwidth between to compute units and memory can be advantageous in the training process, infact bandwidth is a critical bottleneck across several training workloads.

Low latency is more critical to real world inference applications. Unlike in training, during inference it is more common to process just or very few samples at a time, rather than a large batch of samples to meet the latency requirements of the application. This means for Inference the matrix vector multiplication are more common than GEMMs. Matrix vector multiplication still benefits from parallelism but have less parallelism than general matrix multiplications. Deep Learning applications used to be primarily around image classification nd other visual tasks. Vision remains largest workload in edge applications such as security camera, autonomous vehicles. Recommendation and language tasks dominate across the companies with the largest global data centers. Neural Networks for language and recommendation system are not only growing in number of weights but also in their complexity. 

Trends in new Neural Network models
===
Traditional neural network models comprises sequential layers and the inputs and outputs are of static sizes. Newer neural networks have dynamic inputs and outputs sizes and are adopting a cross structure where the outputs of several non-sequential layer can become input to another layer. The computational pattern and memory accesses are predictable for traditional neural networks as they required a lot of MACs i.e. Multiply and Addition operations and therefore required high memory bandwidth for training. The predictable nature of computation patterns made it simpler for the software to pipeline the data to efficiently use the hardware. 

Newer Neural networks have more irregular memory access patterns and in addition to MACs and bandwidth they also require larger memory capacity, high operating frequency for the growing number of non parallelize computations and lower memory read and write latencies across non-sequential memory addresses. Memory and bandwidth are becoming the central components in deep learning system designs. Hardware managed memory or caches consume more power whereas software managed memory or scratchpads typically used in dedicated AI processors or ASICs increase the software complexity.


The continued exponential growth in MACs for silicon area, the memory bandwidth and the bandwidth nodes within a server and across servers are becoming the limiting factors to improve performance. One way to reduce memory and bandwidth consumption is to use less bits to represent numbers, lower numerical precision reduces the memory and bandwidth bottlenecks. On the other hand it can result in lower accuracy, hence one needs to consider the trade-offs from the computational gains with possible accuracy impact. 

