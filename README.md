# LLMs Foundations and Embeddings

A hands-on introduction to the core mechanics of large language models and vector embeddings. You will build mental models for how LLMs generate text, implement token counting and cost estimation from scratch, compute similarity with pure Python, and assemble a local retrieval pipeline without any external ML libraries.

## Learning Objectives

By the end of this repository, you should be able to:

- Trace how input text is mapped to tokens, embeddings, contextual representations, logits, and generated outputs.
- Explain decoding settings (temperature, top-p) and how they control generation diversity and randomness.
- Count prompt tokens and estimate context-window usage and API consumption costs.
- Compute local text embeddings and determine vector similarity using cosine similarity.
- Build a local sparse retrieval pipeline over custom datasets and explain how it differs from dense semantic search.
- Implement confidence thresholds to evaluate retrieval reliability and mitigate hallucination, prompt injection, and bias.

## Learning Path

| File / Folder | Description |
|---|---|
| [**01 - LLM Foundations**](01-llm-foundations.ipynb) | Trace how next-token prediction, transformer blocks, context windows, and decoding settings influence generated text. |
| [**02 - Tokenization and Cost**](02-tokenization-and-cost.ipynb) | Implement token counting algorithms, estimate API query costs, and plan prompt budgets. |
| [**03 - Embeddings and Similarity**](03-embeddings-and-similarity.ipynb) | Build text vectors locally and compare semantic relatedness using cosine similarity. |
| [**04 - Local Retrieval Mini-Project**](04-local-retrieval-mini-project.ipynb) | Retrieve top-k module candidates with a transparent sparse index and implement confidence thresholds. |
| [**05 - Limitations and Evaluation**](05-limitations-and-evaluation.ipynb) | Identify typical failure modes such as hallucination, bias, prompt injection, and retrieval errors. |
| [**06 - Guided Support Router Exercise**](06-guided-support-router-exercise.ipynb) | Build an end-to-end support router combining token estimation, cost calculation, local retrieval, and risk thresholding. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**assets**](assets/) | Visual aids referenced in the repository. |
| [**data**](data/) | CSV datasets used across the notebooks for embedding and similarity exercises. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a
  **placeholder**. Replace it, including the `< >` brackets, with your own
  value. For example, `cd <repo-name>` becomes `cd mle-llms-foundations-embeddings`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Repository in VS Code

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## References & Further Reading

- [**Attention Is All You Need (Vaswani et al., 2017)**](https://arxiv.org/abs/1706.03762): The foundational paper introducing the transformer architecture and self-attention.
- [**The Illustrated Transformer**](https://jalammar.github.io/illustrated-transformer/): A visual, step-by-step walkthrough of how transformers work, widely used in university courses.
- [**Hugging Face Tokenizers Guide**](https://huggingface.co/docs/tokenizers/index): Explanation and implementation of modern subword algorithms including BPE and WordPiece.
- [**OpenAI Embeddings Guide**](https://developers.openai.com/api/docs/guides/embeddings): Best practices for generating and using semantic embedding vectors via the OpenAI API.
- [**Embeddings: What they are and why they matter**](https://simonwillison.net/2023/Oct/23/embeddings/): A practical introduction to embeddings covering semantic search, RAG, and real-world use cases with SQLite.
