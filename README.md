# 🧠 Natural Language Understanding (NLU) - Named Entity Recognition (NER)

## 📚 Project Overview
This project focuses on **Natural Language Understanding (NLU)** using **BERT-based Named Entity Recognition (NER)**. It trains a model to identify entities in text and includes a chatbot for interactive NER.

✅ **Uses BERT (bert-base-uncased) for token classification**  

✅ **Processes textual data for Named Entity Recognition (NER)**  

✅ **Trains an NER model using Hugging Face's Transformers**  

✅ **Interactive chatbot for Named Entity Recognition (NER)**  

---

## 📖 Table of Contents
- [📚 Project Overview](#-project-overview)
- [🎯 Key Features](#-key-features)
- [⚙️ Installation & Dependencies](#-installation--dependencies)
- [🚀 Usage Instructions](#-usage-instructions)
- [🗂️ Dataset Information](#-dataset-information)
- [🤖 NER Chatbot](#-ner-chatbot)
- [🧩 Contributing](#-contributing)
- [📩 Contact & Support](#-contact--support)
- [📜 License](#-license)

---

## 🎯 Key Features
- **BERT-based Named Entity Recognition (NER) Model**
- **Processes textual datasets and tokenizes data**
- **Custom training pipeline for entity classification**
- **Simple chatbot to detect entities from user input**

---

## ⚙️ Installation & Dependencies
To set up the project, install the required dependencies:

```sh
# Clone the repository
git clone https://github.com/hamayl001/NLU_NER_Project.git
cd NLU_NER_Project

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage Instructions
Run the following commands to train and interact with the model:

```sh
# Train the NER model
python train.py

# Run the interactive NER chatbot
python chatbot.py
```

---

## 🗂️ Dataset Information
- **Dataset Used:** NER dataset (`ner_dataset_fixed.csv`)

- **Format:** Each row contains `word`, `sentence_id`, and `tag`.
- 
- **Example Entities Recognized:**
- 
  - `B-PER` (Person Names)
  - `B-LOC` (Locations)
  - `B-ORG` (Organizations)

---

## 🤖 NER Chatbot
An interactive chatbot that identifies **Named Entities** from user input.

```sh
# Run the chatbot
python chatbot.py
```

Example:
```sh
You: Barack Obama was the 44th president of the USA.
Bot: Entities: [('Barack Obama', 'PER'), ('USA', 'LOC')]
```

---

## 🧩 Contributing

### 🔗 How to Contribute
```sh
# Fork the repository
git fork https://github.com/hamayl001/NLU_NER_Project.git

# Create a new branch
git checkout -b feature-branch

# Commit your changes
git commit -m "Added new feature"

# Push to GitHub
git push origin feature-branch

# Submit a Pull Request
```

---

## 📩 Contact & Support

### 📧 Contact Information

- **Email:** [maylzahid588@gmail.com](mailto:maylzahid588@gmail.com)

🤝 **Open to collaboration and improvements!**

---

## 📜 License

This project is licensed under the **MIT License**.

---

✅ **Project Status:** Completed  
**by #Hamayl Zahid**


