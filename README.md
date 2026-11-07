# Generative-AI

Generative AI experiments: a Streamlit PDF chatbot and Colab notebooks.

## PDF chatbot

`google gemini LLM/llm_chat.py` reads uploaded PDFs with PyPDF2 and answers questions with the Gemini API.

Streamlit 1.67.0 requires Python 3.11 or newer. Continuous integration runs Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run "google gemini LLM/llm_chat.py"
```

The script sets `GEMINI_API_KEY` to the placeholder `use_your_api_keys`. Replace that placeholder before asking a question. The Gemini request runs only after PDFs are uploaded and a question is submitted.

Direct dependencies are pinned in `google gemini LLM/requirements.txt`. The root `requirements.txt` installs that file. The `google-generativeai` package provides `import google.generativeai`. `PyPDF2` and `google-generativeai` are deprecated by their publishers; `llm_chat.py` still imports them, so this setup keeps those libraries.

## Checks

```bash
python -m pip install -r requirements-dev.txt
flake8 .
pytest
```

Lint and tests cover the Python modules and their dependency declarations. They do not execute notebooks.

## Notebooks

These notebooks stay on their current paths and install libraries in their own cells:

- `LLM Agents/Create_vector_database.ipynb`
- `LLM Agents/Interview_questions_genarator.ipynb`
- `LLM Agents/interview_que_duck_duck_go.ipynb`
- `google gemini LLM/Google_gemini_hugging_face.ipynb`
- `docs/docs/examples/finetuning/embeddings/finetune_embedding.ipynb`

## Open-source LLM

`Open Source LLM/mistral_llm.py` is a placeholder module. It does not load a model.
