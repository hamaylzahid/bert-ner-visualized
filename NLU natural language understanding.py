import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np

# Load the dataset
df = pd.read_csv("C:/Users/user/Downloads/ner_dataset_fixed.csv")

# Convert words and tags into grouped sentences
sentences, labels = [ ], [ ]
current_sentence, current_labels = [ ], [ ]
current_id = df.iloc[ 0 ][ 'sentence_id' ]

for _, row in df.iterrows():
    if row[ 'sentence_id' ] != current_id:
        sentences.append(current_sentence)
        labels.append(current_labels)
        current_sentence, current_labels = [ ], [ ]
        current_id = row[ 'sentence_id' ]

    current_sentence.append(row[ 'word' ])
    current_labels.append(row[ 'tag' ])

# Append the last sentence
if current_sentence:
    sentences.append(current_sentence)
    labels.append(current_labels)

# Tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

# Define label mappings
unique_tags = sorted(set(df[ 'tag' ]))  # Ensure consistent order
tag2id = {tag: idx for idx, tag in enumerate(unique_tags)}
id2tag = {idx: tag for tag, idx in tag2id.items()}


# Tokenize & align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples[ "tokens" ], truncation=True, padding="max_length", max_length=128,
                                 is_split_into_words=True)
    labels = [ ]
    for i, word_ids in enumerate(
            tokenized_inputs.word_ids(batch_index=batch_idx) for batch_idx in range(len(examples[ "tokens" ]))):
        label_ids = [ ]
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(tag2id.get(examples[ "ner_tags" ][ i ][ word_id ], -100))
        labels.append(label_ids)
    tokenized_inputs[ "labels" ] = labels
    return tokenized_inputs


# Convert data to Hugging Face dataset format
dataset = Dataset.from_dict({
    "tokens": sentences,
    "ner_tags": labels
})
dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Split into train and test
dataset = dataset.train_test_split(test_size=0.2)

# Define model
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(unique_tags))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset[ "train" ],
    eval_dataset=dataset[ "test" ],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")

print("Model training complete! Model saved in './ner_model'")


# Evaluation Function
def evaluate_model():
    predictions, labels, _ = trainer.predict(dataset[ "test" ])
    predictions = np.argmax(predictions, axis=2)
    true_labels = [ [ id2tag[ label ] for label in sent_labels if label != -100 ] for sent_labels in labels ]
    pred_labels = [ [ id2tag[ pred ] for pred, label in zip(sent_preds, sent_labels) if label != -100 ] for
                    sent_preds, sent_labels in zip(predictions, labels) ]

    flat_true = sum(true_labels, [ ])
    flat_pred = sum(pred_labels, [ ])

    print("Classification Report:")
    print(classification_report(flat_true, flat_pred))
    print("Accuracy:", accuracy_score(flat_true, flat_pred))
    print("F1 Score:", f1_score(flat_true, flat_pred, average='weighted'))


# Run Evaluation
evaluate_model()


# Simple Chatbot for NER


def ner_chatbot():
    print("NER Chatbot Initialized! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Tokenize user input
        tokens = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**tokens)

        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
        words = tokenizer.convert_ids_to_tokens(tokens[ "input_ids" ].squeeze().tolist())

        # Process results, merge subwords
        final_entities = [ ]
        current_entity = ""
        current_label = "O"

        for word, pred in zip(words, predictions):
            if word in [ "[CLS]", "[SEP]" ]:  # Ignore special tokens
                continue

            word = word.replace("##", "")  # Merge subwords

            entity_label = id2tag.get(pred, "O")

            if entity_label.startswith("B-"):  # Start of an entity
                if current_entity:  # Save previous entity
                    final_entities.append((current_entity, current_label))
                current_entity = word
                current_label = entity_label[ 2: ]  # Extract entity type (PER, LOC, ORG, etc.)
            elif entity_label.startswith("I-") and current_entity:  # Continuation of entity
                current_entity += " " + word
            else:  # No entity
                if current_entity:
                    final_entities.append((current_entity, current_label))
                current_entity = ""
                current_label = "O"

        # Save last detected entity
        if current_entity:
            final_entities.append((current_entity, current_label))

        # Display detected entities
        if final_entities:
            print("Entities:", final_entities)
        else:
            print("No entities detected.")


# Run the chatbot
ner_chatbot()
