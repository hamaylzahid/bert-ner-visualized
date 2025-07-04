<!-- Banner -->
<br><h1 align="center">🧠 BERT-based Named Entity Recognition (NER)</h1><br>
<p align="center">
  <b>Natural Language Understanding using Transformer Models</b><br>
  <i>Train, evaluate, and interact with an NER model — with clean visualizations and chatbot support.</i>
</p>

<p align="center">
  <!-- Badges -->
<p align="left">
  <img src="https://img.shields.io/github/languages/top/hamaylzahid/bert-ner-visualized?color=blueviolet" />
  <img src="https://img.shields.io/github/license/hamaylzahid/bert-ner-visualized?style=flat-square" />
  <img src="https://img.shields.io/github/last-commit/hamaylzahid/bert-ner-visualized?color=green" />
  <img src="https://img.shields.io/badge/Model-BERT--base--uncased-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Chatbot-Interactive-orange?style=flat-square" />
</p>

---

<br><h2 align="center">📖 Table of Contents</h2><br>

- [🧠 Project Overview](#-overview)  
- [📁 Dataset](#-dataset)
- [📥 Dataset Format](#-dataset-format)
- [🚀 Features](#-features)  
- [⚙️ Setup Instructions](#️-setup-instructions)  
- [🏃 Run Training](#-run-training)  
- [📊 Evaluation Metrics](#-evaluation-metrics)    
- [🤖 NER Chatbot](#-ner-chatbot)   
- [🤝 Contact & Contribution](#-contact--contribution)  
- [📜 License](#-license)

---

<br><h2 align="center">📌 Overview</h2><br>

This project demonstrates a full pipeline for **Named Entity Recognition (NER)** using **BERT (`bert-base-uncased`)**, built for CPU compatibility. It includes:

- 🔤 Token-level classification for `PER`, `ORG`, `LOC`, etc.
- 📊 Evaluation metrics (F1, Accuracy, Classification Report)
- 📉 Visualizations for model performance
- 💬 NER Chatbot for real-time entity detection
- 🧠 Based on HuggingFace Transformers and PyTorch

---

<br><h2 align="center">📁 Dataset</h2><br>

- **Source**: Custom dataset
- **Format**: CSV with 3 columns:
  - `sentence_id`
  - `word`
  - `tag` (BIO format: `B-PER`, `I-PER`, `O`, etc.)

```plaintext
sentence_id, word,    tag
1,           Elon,    B-PER
1,           Musk,    I-PER
1,           founded, O
```
# 🤖 Named Entity Recognition (NER) using BERT

A complete CPU-optimized BERT-based Named Entity Recognition (NER) pipeline using Hugging Face Transformers and Datasets. It includes model training, evaluation, visualizations, and an interactive NER chatbot.

---

<br><h2 align="center">📥 Update Dataset Path</h2><br>

Change the path in `main.py` if needed:

```python
df = pd.read_csv("C:/Users/you/path/to/ner_dataset_fixed.csv")

```
# 🤖 Named Entity Recognition (NER) using BERT

A complete CPU-optimized BERT-based NER system using Hugging Face Transformers. Includes training, evaluation, plots, and an interactive chatbot.

```

```
<br><h2 align="center">📥 Dataset Format</h2><br>


sentence_id | word   | tag
------------|--------|-----
1           | Elon   | B-PER
1           | Musk   | I-PER
1           | founded| O
2           | Google | B-ORG
...
```

```
🛠 Update the dataset path in code:

df = pd.read_csv("C:/Users/your/path/to/ner_dataset_fixed.csv")
```

```
<br><h2 align="center">🚀 Features</h2><br>

✅ BERT Token Classification with HuggingFace  
✅ Preprocessing + Tag Alignment  
✅ Trainer API for CPU training  
✅ Classification Report & Weighted F1  
✅ Visual Plots: Loss Curve, Confusion Matrix, Entity Distribution  
✅ CLI Chatbot for live NER extraction
```

```
<br><h2 align="center">⚙️ Setup Instructions</h2><br>

# Clone the repository
git clone https://github.com/yourusername/BERT-NER.git
cd BERT-NER

# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install pandas scikit-learn matplotlib seaborn transformers datasets
```

```
<br><h2 align="center">🏃 Run Training</h2><br>


python main.py
```

```
<br><h2 align="center">📊 Evaluation Metrics</h2><br>

- Accuracy  
- Weighted F1 Score  
- Classification Report (Precision, Recall, F1)
```



```
<br><h2 align="center">🤖 NER Chatbot</h2><br>

# Run the chatbot
python main.py

# Example:
You: Barack Obama was the 44th president of the USA.  
Bot: Entities: [('Barack Obama', 'PER'), ('USA', 'LOC')]

Type 'exit' to quit.

<br><h2 align="center">🤝 Contact & Contribution</h2><br>

Have feedback, ideas, or want to collaborate on improving this NER project?

- 📧 **Email**: [maylzahid588@gmail.com](mailto:maylzahid588@gmail.com)  
- 🌟 Star this repo to support the work  
- 🤝 Fork and contribute—PRs are always welcome!

---

<br><h2 align="center">📜 License</h2><br>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://github.com/hamaylzahid/bert-ner-visualized/commits/main"><img src="https://img.shields.io/github/last-commit/hamaylzahid/bert-ner-visualized?color=blue" alt="Last Commit"></a>
  <a href="https://github.com/hamaylzahid/bert-ner-visualized"><img src="https://img.shields.io/github/repo-size/hamaylzahid/bert-ner-visualized?color=lightgrey" alt="Repo Size"></a>
</p>

This project is licensed under the **MIT License** – feel free to use, modify, and distribute.

**✅ Project Status:** Completed and production-ready  
**🧾 License:** MIT – [View Full License »](LICENSE)

---

<br><br>

<p align="center" style="font-family:Segoe UI, sans-serif;">
  <img src="https://img.shields.io/badge/Built%20with-Python-blue?style=flat-square&logo=python&logoColor=white" alt="Python Badge" />
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-ffcc00?style=flat-square&logo=huggingface&logoColor=black" alt="Transformers Badge" />
</p>

<p align="center">
  <b>Crafted for real-world NER tasks — lightweight, explainable, and fully visualized</b> ✨
</p>

<p align="center">
  <a href="https://github.com/hamaylzahid">
    <img src="https://img.shields.io/badge/GitHub-%40hamaylzahid-181717?style=flat-square&logo=github" alt="GitHub" />
  </a>
  •  
  <a href="mailto:maylzahid588@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact%20Me-red?style=flat-square&logo=gmail&logoColor=white" alt="Email Badge" />
  </a>
  •  
  <a href="https://github.com/hamaylzahid/bert-ner-visualized">
    <img src="https://img.shields.io/badge/Repo-Link-blueviolet?style=flat-square&logo=github" alt="Repo" />
  </a>
  <br>
  <a href="https://github.com/hamaylzahid/bert-ner-visualized/fork">
    <img src="https://img.shields.io/badge/Fork%20This%20Project-Clone%20and%20Build-2ea44f?style=flat-square&logo=github" alt="Fork Project Badge" />
  </a>
</p>

<p align="center">
  <sub><i>Build responsibly. Deploy transparently. Recognize intelligently.</i></sub>
</p>

<p align="center">
  🧠 <b>Use this project to demonstrate your skills in NLP, Transformers, and Explainable AI</b>  
  <br>
  🚀 Clone it, explore it, refine it — and share your improvements with the community.
</p>


