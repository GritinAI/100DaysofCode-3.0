  
Day 26

RAG Indexing and Retrieval

Welcome to Day 26 of the 100 Days of Code challenge\!

Today, we’ll learn more about RAG (Retrieval-Augmented Generation)—an advanced technique that combines retrieval methods with generative models for more accurate and informed responses.

Learn and master Rag Indexing and Retrieval techniques to efficiently store, search, and retrieve data

Key Topics for Today:

[RAG Indexing](https://youtu.be/bjb_EMsTDKI?si=w-KesbdFjiNkdDgJ): Learn how to index data effectively to improve retrieval for generative models.  
[RAG Retrieval](https://youtu.be/LxNVgdIz9sU?si=AcO729_jZ2DdZCWW): Explore how relevant information is retrieved from indexed data to enhance the output of AI models.

---

Day 27

RAG Generation 

Welcome to Day 27 of the 100 Days of Code challenge\!

Today, we’ll learn how RAG produces informative and engaging text, thus making it a powerful tool for various NLP tasks.

Learn more [here](https://youtu.be/Vw52xyyFsB8?si=fGF0Fp3GMZgLj_N0)

---

Day 28

RAG Multi-query

Welcome to Day 28 of the 100 Days of Code challenge\!

Today you will learn to handle multiple queries simultaneously and retrieve relevant information from external knowledge sources to generate contextually appropriate responses for each query. 

Get [Started](https://youtu.be/JChPi0CRnDY?si=oNStN1DV8v-yhXCT) 

---

Day 29

RAG Fusion

Welcome to Day 29 of the 100 Days of Code challenge\!

Today you will learn to combine the knowledge retrieval capabilities of RAG with the text fusion capabilities of Fusion models to generate high-quality, coherent, and accurate text. 

Explore RAG Fusion [here](https://youtu.be/77qELPbNgxA?si=qVfarlP-WxHp4hfP)

---

Day 30

Decomposition  
   
Welcome to Day 30 of the 100 Days of Code challenge\!

Today you will learn about Query decomposition. This is a strategy used to improve question-answering by breaking down a question into sub-questions. These can either be (1) solved sequentially or (2) independently answered followed by consolidation into a final answer. 

Learn more about [query decomposition](https://youtu.be/77qELPbNgxA?si=qVfarlP-WxHp4hfP) 

---

Day 31

RAG Intermediate \- Step Back 

Welcome to Day 31 of the 100 Days of Code challenge\!

Today we will learn about Step-back prompting which is an approach to improve retrieval that builds on chain-of-thought reasoning. From a question, it generates a step-back (higher level, more abstract) question that can serve as a precondition to correctly answering the original question. This is especially useful in cases where background knowledge or more fundamental understanding is helpful to answer a specific question.

Learn more [here](https://youtu.be/xn1jEjRyJ2U?si=z5dr-2pyVCGP8qZk)

---

Day 32 

RAG Intermediate \- Hyde

Welcome to Day 32 of the 100 Days of Code challenge\!

Today we will learn about HyDE\!

HyDE (Hypothetical Document Embeddings) is an approach to improve retrieval that generates hypothetical documents that could be used to answer the user input question. These documents, drawn from the LLMs knowledge, are embedded and used to retrieve documents from an index. The idea is that hypothetical documents….learn [more](https://youtu.be/SaDzIVkYqyY?si=WRIMJ1qhE3xteD-C)

---

Day 33

RAG Intermediate \- Routing 

Welcome to Day 33 of the 100 Days of Code challenge\!

Today we will focus on the different types of [query routing](https://youtu.be/pfpIndq7Fi8?si=uZfYTBiqA4MGTDLX) 

1. Logical   
2. Semantic

Routing is basically routing potentially different question to the right source, in many cases it might be a different database. learn more [here](https://youtu.be/pfpIndq7Fi8?si=uZfYTBiqA4MGTDLX)

---

Day 34 

Query Structuring 

Welcome to Day 34 of the 100 Days of Code challenge\!

RAG systems ingest questions in natural language but we interact w/ databases using domain-specific languages (e.g., SQL, Cypher for Relational and Graph DBs). And, many vectorstores have metadata that can allow for structured queries to filter chunks. Let’s [dive](https://youtu.be/kl6NwWYxvbM?si=9cekJCKgMj_tUhHk) into how user questions are converted into structured queries.  
---

Day 35 

Multi-Representation Indexing

Welcome to Day 35 of the 100 Days of Code challenge\!

Many RAG approaches focus on splitting documents into chunks and returning some number upon retrieval for the LLM. But chunk size and chunk number can be brittle parameters that many users find difficult to set; both can significantly affect results if they do not contain all context to answer a question.

Today we will learn some useful tricks for indexing full documents. Check it [out](https://youtu.be/gTCU9I6QqCE?si=B2hKU80eVBYfJARR).

---

Day 36

Deep Dive into RAG \- Raptor

Welcome to Day 36 of the 100 Days of Code challenge\!

Today we will learn a technique used in RAG (Retrieval-Augmented Generation) systems to handle both "lower-level" and "higher-level" questions. Raptor is a sparse retrieval method that enables the model to retrieve relevant documents and passages across a large corpus, even if they are not among the top-k nearest neighbors.

Raptor is particularly useful for handling "higher-level" questions that require distilling ideas across multiple documents. By using Raptor, RAG systems can retrieve a larger set of relevant documents and passages, allowing the model to generate more accurate and informative responses….learn [more](https://youtu.be/z_6EeA2LDSw?si=MK9ehP947BKkRkRZ)

---

Day 37

Deep Dive into RAG \- Colbert

Welcome to Day 37 of the 100 Days of Code challenge

Another technique used in RAG systems is  ColBERT (COLlar-based BERT), it is used to improve the efficiency and effectiveness of retrieval. ColBERT uses a late-interaction approach, where the query and documents are represented as collections of embeddings, and the similarity is computed between these collections….learn this technique [here](https://youtu.be/cN6S0Ehm7_8?si=GU_kiR1klr_H7NFD)

---

Day 38

Local Agentic RAG with llama3

Welcome to Day 38 of the 100 Days of Code challenge  
Local Agentic RAG is a variant of the RAG architecture that focuses on using local knowledge to generate responses. By incorporating LLaMA3, a large language model, into the Local Agentic RAG framework, you can leverage the strengths of both models to generate high-quality responses….let’s explore how it works [here](https://www.youtube.com/watch?v=u5Vcrwpzoz8)

---

Day 39

Road to LLM finetunning

Welcome to Day 39 of the 100 Days of Code challenge  
Today we will learn about finetuning LLM. The road to LLM finetuning involves pretraining a large language model, preparing a dataset, selecting and configuring the model, finetuning, evaluating, and iterating to optimize performance on a specific task.

Key Topic for Today:

[Quantization](https://youtu.be/6S59Y0ckTm4?si=J28Jb5hFcWlInQLP): This is a common technique used to reduce the model size, though it can sometimes result in reduced accuracy.  
---

Day 40

Mathematical Intuition 

Welcome to Day 40 of the 100 Days of Code challenge\!

Today we will be learning about some finetuning techniques which is called as LoRA and QLoRA techniques, low order rank adaptation, and quantized Lora.

Let’s get [started](https://youtu.be/l5a_uKnbEr4?si=DZEQ_wQETCsNSn-U)\!

---

Day 41

1-bit LLMs \- The Era of 1-bit 

Welcome to Day 41 of the 100 Days of Code challenge\!

1-bit LLMs (Large Language Models) refer to a new generation of language models that use 1-bit precision (binary) weights and activations, instead of the traditional 32-bit floating-point precision.

What are the implications of this innovation? Find out more [here](https://youtu.be/wN07Wwtp6LE?si=QaId_RC0CbZWGbkq)

---

Day 42

Simplify LLMOps & Build LLM pipeline in Minutes

Welcome to Day 42 of the 100 Days of Code challenge\!

Today, we will learn how to quickly build and deploy an LLM pipeline. LLMOps (Large Language Model Operations) simplifies the deployment and management of large language models. To build an LLM pipeline in minutes, watch this [video](https://youtu.be/4ijnajzwor8?si=lDeWNZvt_p65CMon).  
---

Day 43

Llama 2 with Custom Dataset

Welcome to Day 43 of the 100 Days of Code challenge\!

Today we will be discussing how we can fine tune LLAMA 2 model with custom dataset using parameter efficient Transfer Learning  using LoRA :Low-Rank Adaptation of Large Language Models.

Note that by fine-tuning LLaMA 2 on your custom dataset, you can create a highly specialized language model that excels in your specific use case. Check it [out](https://youtu.be/Vg3dS-NLUT4?si=YoVlwnCyIv3Xm-c8)  
---

Day 44

Finetune Gemma models 

Welcome to Day 44 of the 100 Days of Code challenge\!

Today we will explore LoRA for finetuning Gemma (via Keras framework). LoRA is a technique for adapting pre-trained models to specific tasks with minimal additional parameters. With LoRA, you can adapt Gemma to your specific task with minimal additional parameters, making it more efficient and effective. LoRA is particularly useful when you have a small to medium-sized dataset and want to leverage the pre-trained model's knowledge.

Learn how to use [LoRA](https://youtu.be/IZXNgu4dW70?si=mlRawnm8pOOSOBaQ).  
---

Day 45

Gen AI Fine Tune LLM model \- Model Crash Course

Welcome to Day 45 of the 100 Days of Code challenge\!

Today, we are going to learn how we can fine-tune LLM in 2024\.  It’s no secret that [large language models (LLMs)](https://www.superannotate.com/blog/llm-overview) are evolving at a wild speed and are turning heads in the [generative AI](https://www.superannotate.com/blog/generative-ai-explained) industry. Enterprises aren't just intrigued; they're obsessed with LLMs, particularly. [Continue reading](https://www.superannotate.com/blog/llm-fine-tuning) 

---

Day 46

LLM Agent 1 \- Deep Learning Short Course 

In this course you will learn to build an agent from scratch using Python and an LLM, and then you will rebuild it using LangGraph, learning about  its components and how to combine them to build flow-based applications.

Enroll [now](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/).  
---

Day 47 

AI Agentic Design Patterns with AutoGen

Welcome to Day 47 of the 100 Days of Code challenge\!

Today we will learn about AutoGen. This is a tool that generates code for AI models, and when combined with AI Agentic Design Patterns, it enables the creation of autonomous AI systems that can adapt and learn. Learn more about [AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/).

---

Day 48

Architecting & Testing reliable agents

Welcome to Day 48 of the 100 Days of Code challenge\!

Architecting and testing reliable agents is crucial for ensuring that autonomous systems operate correctly and make informed decisions. 

The best practices includes:

* Use established development frameworks and libraries  
* Implement robust error handling and exception handling  
* Conduct regular code reviews and testing  
* Utilize continuous integration and deployment pipelines  
* Monitor agent performance and update as needed

Check out the [slide](https://docs.google.com/presentation/d/1QWkXi4DYjfw94eHcy9RMLqpQdJtS2C_kx_u7wAUvlZE/edit#slide=id.g273e7f400bc_0_0) to learn more

---

Day 49

Breakout Session: LLM Agent

Welcome to Day 49 of the 100 Days of Code challenge\!

Today we will be having a breakout session to discuss all we have learnt so far, trash out any questions or misunderstandings you might have. 

In anticipation of today’s meet, read up on [LLM Agents](https://www.promptingguide.ai/research/llm-agents) 

---

Day 50

Learning Tensorflow I

Welcome to Day 50 of the 100 Days of Code challenge\!

We are digging deeper. Today we will be exploring Tensorflow. TensorFlow allows researchers and developers to easily scale up their machine learning models and train them on large datasets, making it a crucial tool for large-scale deep learning tasks.

Are you ready to learn the fundamentals of TensorFlow and deep learning with Python? Follow this tutorial to have a good grasp of [tensorflow](https://www.youtube.com/watch?v=tpCFfeUEGs8).

---

Day 51

Learning Tensorflow II

Welcome to Day 51 of the 100 Days of Code challenge\!

Today we are going to continue from where we left off yesterday. Don’t forget that with tensorflow you can:

* Train models on multiple GPUs or machines  
* Handle large datasets that don't fit in memory  
* Speed up training processes

Check out [Tensorflow II](https://www.youtube.com/watch?v=ZUKz4125WNI&t=0s)

---

Day 52

Learning PyTorch A

Welcome to Day 52 of the 100 Days of Code challenge\!

PyTorch is a library that has gained significant popularity in the research community, particularly in areas like computer vision, natural language processing, and reinforcement learning. Its dynamic computation graph and flexibility make it an attractive choice for many researchers and developers. 

Sit tight, let's learn [Pytorch](https://www.youtube.com/watch?v=Z_ikDlimN6A).

---

Day 53

Learning PyTorch A

Welcome to Day 53 of the 100 Days of Code challenge\!

You probably did a speed watch of yesterday’s video on Pytorch. Let’s break it down into four sections.

Today, we will go over the first 6 hours of the video.

Rewatch [here](https://www.youtube.com/watch?v=Z_ikDlimN6A)

---

Day 54

Learning PyTorch A

Welcome to Day 54 of the 100 Days of Code challenge\!

Today, we will explore the next 6 hours from where we stopped yesterday. We will continue to discuss how we can set up loss functions, optimizers and improve a model.

Rewatch [here](https://www.youtube.com/watch?v=Z_ikDlimN6A&t=21716s)

---

Day 55

Learning PyTorch A

Welcome to Day 55 of the 100 Days of Code challenge\!

Today is our 4th day on Pytorch, we will discuss:

1. Troubleshooting a multiclass model  
2. Introduction to computer vision  
3. Creating a train/test loop   
4. Coding in CNN

Start from 12:00:00 [here](https://www.youtube.com/watch?v=Z_ikDlimN6A&t=42658s)

---

Day 56

Learning PyTorch A

Welcome to Day 56 of the 100 Days of Code challenge\!

Today we will complete the review of learning Pytorch A. The last 6 hours of the video covers the entire process of working with custom datasets in deep learning, from downloading and exploring the data to building and training a model, and finally predicting on new data. It also covers important concepts like data augmentation, overfitting, and underfitting.

Resume [here](https://www.youtube.com/watch?v=Z_ikDlimN6A&t=66514s)

---

Day 57

Learning PyTorch B

Welcome to Day 57 of the 100 Days of Code challenge\!

Today we will learn how to build deep learning models with PyTorch and Python. Let’s make PyTorch a bit more approachable for people starting out with deep learning and neural networks. 

We will focus on the first 4 hours of [PyTorch](https://www.youtube.com/watch?v=GIsg-ZUy0MY) and learn:

* PyTorch Basics & Linear Regression  
* Image Classification with Logistic Regression  
* Training Deep Neural Networks on a GPU with PyTorch

---

Day 58

Learning PyTorch B

Welcome to Day 58 of the 100 Days of Code challenge\!

Yesterday, we were able to focus on the first part of [PyTorch](https://www.youtube.com/watch?v=GIsg-ZUy0MY). Today we will continue from [4:44:51](https://www.youtube.com/watch?v=GIsg-ZUy0MY&t=17091s) and explore the following concepts:

* Image Classification using Convolutional Neural Networks  
* Residual Networks, Data Augmentation and Regularization  
* Training Generative Adverserial Networks (GANs)

At the end of this tutorial, you should have a good grasp of PyTorch. 

Have fun learning\!

---

Day 59

Tensorflow vs PyTorch

Welcome to Day 59 of the 100 Days of Code challenge\!

Over the past few days, we have explored Tensorflow and PyTorch. Now, let’s look into the interplay and disparities of the two frameworks.

Check it out [here](https://www.youtube.com/watch?v=4L86D_fU6sQ)