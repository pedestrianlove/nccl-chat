# Chatbot of NCCL documentation

- Included installation/user guides.

## 0. Setup the environment
```bash
conda env create -f scrape_env.yaml
conda activate scrape_env
```

## 1. Scrape the docs and build ChromaDB.
```bash
sh scrape.sh
```

## 2. Spin up the chatbot.
```bash
python chat.py
```

## References
- https://tomstechacademy.com/build-a-chatbot-with-rag-retrieval-augmented-generation/
- https://github.com/jerpint/RAGTheDocs
