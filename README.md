<!-- Banner -->
<p align="center">
  <img src="https://raw.githubusercontent.com/hamaylzahid/bert-ner-visualized/refs/heads/main/banner_bert.png" alt="BERT NER Banner" style="width:100%; max-width:900px;" />
</p>

<br><h1 align="center">🧠 BERT-based Named Entity Recognition (NER)</h1><br>
<p align="center">
  <b>Natural Language Understanding using Transformer Models</b><br>
  <i>Train, evaluate, and interact with an NER model — with clean visualizations and chatbot support.</i>
</p>


<p align="center">
  <!-- Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/github/languages/top/hamaylzahid/bert-ner-visualized?color=blueviolet" />
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

---


<br><h2 align="center">🤖 Named Entity Recognition (NER) using BERT</h2><br>

<p>
  A complete CPU-optimized BERT-based Named Entity Recognition (NER) pipeline using Hugging Face Transformers and Datasets.<br>
  Includes model training, evaluation, clean visualizations, and an interactive CLI-based chatbot.
</p>

---

<br><h2 align="center">📥 Update Dataset Path</h2><br>

Change the path in `main.py` if needed:

```python
df = pd.read_csv("C:/Users/you/path/to/ner_dataset_fixed.csv")

```
<br><h2 align="center">📥 Dataset Format</h2><br>

<p>
  The dataset used for training follows a standard BIO tagging format where each row represents a single token and its corresponding named entity label.
  <br>
  Multiple rows with the same <code>sentence_id</code> belong to the same sentence.
</p>

<p align="center">
  The columns are:
</p>

<ul>
  <li><code>sentence_id</code> — Identifier for grouping words into sentences</li>
  <li><code>word</code> — Individual tokens (words)</li>
  <li><code>tag</code> — Entity labels in BIO format (<code>B-PER</code>, <code>I-PER</code>, <code>O</code>, etc.)</li>
</ul>

<p align="center"><b>Sample Format:</b></p>

<pre align="center"><code>
sentence_id | word    | tag
------------|---------|-----
1           | Elon    | B-PER
1           | Musk    | I-PER
1           | founded | O
2           | Google  | B-ORG
</code></pre>


---

<br><h2 align="center">🚀 Features</h2><br>

<p>
  This project offers a complete pipeline for training and evaluating a Named Entity Recognition (NER) model using BERT. It's tailored for ease of use, CPU compatibility, and insightful output.
</p>

<ul>
  <li>✅ <b>BERT Token Classification</b> using HuggingFace Transformers</li>
  <li>✅ <b>Smart Preprocessing</b> with proper tag alignment (BIO format)</li>
  <li>✅ <b>Trainer API</b> compatible with CPU for smooth training</li>
  <li>✅ <b>Comprehensive Evaluation</b> with classification report & weighted F1 score</li>
  <li>✅ <b>Visual Analysis</b>: Loss Curve, Confusion Matrix, and Entity Distribution</li>
  <li>✅ <b>Interactive CLI Chatbot</b> for real-time named entity extraction</li>
</ul>

<p>
  🧠 Whether you're a beginner or NLP practitioner, this project simplifies NER with BERT — from raw text to insightful results.
</p>


---
<br><h2 align="center">⚙️ Setup Instructions</h2><br>

<br><h4 align="left"> Clone the repository</h4><br>

git clone https://github.com/yourusername/BERT-NER.git
cd BERT-NER

<br><h4 align="left">Install dependencies</h4><br>

pip install -r requirements.txt

<br><h4 align="left">Or install individually</h4><br>

pip install pandas scikit-learn matplotlib seaborn transformers datasets

---

<br><h2 align="center">🏃 Run Training</h2><br>

<p>
  Once the setup is complete and the dataset path is properly configured, you can train the BERT-based NER model by running the main script.
  <br>
  Training is optimized for CPU usage and saves model outputs and logs for analysis.
</p>

<p align="center"><b>Command:</b></p>

<pre align="center"><code>python main.py</code></pre>

<p>
  After execution, model checkpoints and training logs will be saved in the respective folders like <code>ner_model/</code> and <code>results/</code>.
  <br>
  You can visualize performance in the <code>visuals/</code> directory after training completes.
</p>


```bash
python main.py


```
<br><h2 align="center">📊 Evaluation Metrics</h2><br>

<p>
  After training, the model is evaluated using standard classification metrics that reflect its performance on Named Entity Recognition tasks.
  <br>
  These metrics help assess both overall accuracy and how well each entity type is recognized.
</p>

<ul>
  <li><b>Accuracy</b> – Overall percentage of correctly predicted tokens.</li>
  <li><b>Weighted F1 Score</b> – F1-score considering support (number of true instances) for each class.</li>
  <li><b>Classification Report</b> – Includes <code>precision</code>, <code>recall</code>, and <code>F1-score</code> for each entity tag (PER, ORG, LOC, etc.).</li>
</ul>

<p>
  📁 Output is automatically saved to <code>results/</code> and visualized in the <code>visuals/</code> directory.
</p>


  
---


<br><h2 align="center">🤖 NER Chatbot</h2><br>

<br><h4 align="left">Run the chatbot</h4><br>

python main.py

<br><h4 align="left">Example:</h4><br>

You: Barack Obama was the 44th president of the USA.  
Bot: Entities: [('Barack Obama', 'PER'), ('USA', 'LOC')]

Type 'exit' to quit.

<br><h2 align="center">🤝 Contact & Contribution</h2><br>

<p align="center">
  <a href="mailto:maylzahid588@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-Contact%20Me-red?style=for-the-badge&logo=gmail&logoColor=white" alt="Email Badge" />
  </a>
  &nbsp;
  <a href="https://www.linkedin.com/in/hamaylzahid" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge" />
  </a>
</p>

<p>
  Have feedback, ideas, or want to collaborate on improving this NER project?
</p>

<ul>

  <li>🌟 Star this repo to support the work</li>
  <li>🤝 Fork and contribute — PRs are always welcome!</li>
</ul>


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

<p>
  🧠 <b>Use this project to demonstrate your skills in NLP, Transformers, and Explainable AI</b>  
  <br></p>
 <p> 🚀 Clone it, explore it, refine it — and share your improvements with the community.
</p>


