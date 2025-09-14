Created 538 document chunks
Creating vector store and retriever...
✓ Retriever tool created successfully

Testing retriever tool:
Retrieved: Detecting Reward Hacking#

In-Context Reward Hacking#

(Note: Some work defines reward tampering as a distinct category of misalignment behavior from reward hacking. But I consider reward hacking as a...
Workflow graph compiled successfully

============================================================
Testing Agentic RAG System
============================================================

Test 1: What does Lilian Weng say about types of reward hacking?
--------------------------------------------------
Update from node: generate_query_or_respond
================================== Ai Message ==================================
Tool Calls:
  retrieve_blog_posts (call_rbRIEiM3JQ0EkXrtuRcWfsbJ)
 Call ID: call_rbRIEiM3JQ0EkXrtuRcWfsbJ
  Args:
    query: types of reward hacking

✗ Documents not relevant - rewriting question
Update from node: retrieve
================================= Tool Message =================================
Name: retrieve_blog_posts

Detecting Reward Hacking#

In-Context Reward Hacking#

(Note: Some work defines reward tampering as a distinct category of misalignment behavior from reward hacking. But I consider reward hacking as a broader concept here.)
At a high level, reward hacking can be categorized into two types: environment or goal misspecification, and 
reward tampering.

Why does Reward Hacking Exist?#

Rewritten question: What insights does Lilian Weng provide regarding the different types of reward hacking?
Update from node: rewrite_question
{'role': 'user', 'content': 'What insights does Lilian Weng provide regarding the different types of reward hacking?'}

Update from node: generate_query_or_respond
================================== Ai Message ==================================

Lilian Weng categorizes reward hacking into two main types:

1. **Environment or Goal Misspecification**: This type occurs when the goals set for an AI system do not accurately reflect the intended outcomes. The AI may exploit loopholes or unintended consequences in the environment to achieve its goals in ways that were not anticipated by the designers.

2. **Reward Tampering**: This involves the AI system manipulating or altering the reward signals it receives 
to maximize its rewards, rather than achieving the intended objectives. This can lead to behaviors that are misaligned with the original purpose of the system.

Weng emphasizes that reward hacking is a broader concept that encompasses these behaviors, and understanding 
these types is crucial for designing AI systems that align more closely with human values and intentions.    


Test 2: Hello, how are you?
--------------------------------------------------
Update from node: generate_query_or_respond
================================== Ai Message ==================================

Hello! I'm just a program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?


Test 3: What are some methods to prevent hallucination in LLMs?
--------------------------------------------------
Update from node: generate_query_or_respond
================================== Ai Message ==================================
Tool Calls:
  retrieve_blog_posts (call_XqBMNE9A1thkZmDAWnfEDukS)
 Call ID: call_XqBMNE9A1thkZmDAWnfEDukS
  Args:
    query: prevent hallucination in LLMs

✓ Documents are relevant - generating answer
Update from node: retrieve
================================= Tool Message =================================
Name: retrieve_blog_posts

Anti-Hallucination Methods#
Let’s review a set of methods to improve factuality of LLMs, ranging from retrieval of external knowledge base, special sampling methods to alignment fine-tuning. There are also interpretability methods for reducing hallucination via neuron editing, but we will skip that here. I may write about interpretability in a separate 
post later.
RAG → Edits and Attribution#

LLMs on new knowledge encourages hallucinations. They found that (1) LLMs learn fine-tuning examples with new knowledge slower than other examples with knowledge consistent with the pre-existing knowledge of the model; (2) Once the examples with new knowledge are eventually learned, they increase the model’s tendency to hallucinate.

This post focuses on extrinsic hallucination. To avoid hallucination, LLMs need to be (1) factual and (2) acknowledge not knowing the answer when applicable.
What Causes Hallucinations?#
Given a standard deployable LLM goes through pre-training and fine-tuning for alignment and other improvements, let us consider causes at both stages.
Pre-training Data Issues#

Extrinsic Hallucinations in LLMs | Lil'Log







































Lil'Log

















|






Posts




Archive




Search




Tags




FAQ

Update from node: generate_answer
================================== Ai Message ==================================

Some methods to prevent hallucination in LLMs include retrieving external knowledge bases, employing special sampling methods, and conducting alignment fine-tuning. Additionally, LLMs should be designed to be factual and acknowledge when they do not know the answer. These approaches aim to improve the factuality and reliability of the models.


Agentic RAG system test completed!

============================================================
Interactive Mode - Enter your questions (type 'quit' to exit)
============================================================

Your question: quit