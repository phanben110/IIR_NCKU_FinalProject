import os
import streamlit as st  
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import os
from datetime import datetime
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
warnings.filterwarnings("ignore")

device = "cpu"
labels = ["false", "effect", "mechanism", "advise", "int"] 

# Tokenize and encode the sentences
class MyDataset(Dataset):
    def __init__(self, e1, e2, sentence , tokenizer, max_len):
        self.e1 = e1
        self.e2 = e2 
        self.sentence = sentence 
        self.tokenizer = tokenizer 
        self.max_len = max_len

    def __len__(self):
          return 1 

    def __getitem__(self, index):
        if self.sentence == None: 
            input_text = f"{self.e1} [SEP] {self.e2}"
        else:
            input_text = f"{self.sentence} [SEP] {self.e1} [SEP] {self.e2}"

        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': input_text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class BertSentimentClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BertSentimentClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout1(pooled_output)
        output = self.fc1(pooled_output)
        output = self.dropout2(output)
        logits = self.fc2(output)

        return logits

def load_bert_model(weight_path, bert_model_name):
  
    num_classes = 6
    model_predict = BertSentimentClassifier(bert_model_name, num_classes)
    model_predict.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    return model_predict 

def inference():
    st.image("image/4_inference.png")
    st.subheader("Step 1: Choose the BERT architecture and pretraining.", divider='rainbow') 
    choice_model = st.selectbox("Choose the BERT architecture and pretraining model.", 
    ["None","BERT Model 1", "BERT Model 2"]) 
    if choice_model == "BERT Model 1":
        st.image("image/Model1.png")
    elif choice_model == "BERT Model 2": 
        st.image("image/Model2.png")

    choice_pretraining_model = st.selectbox("Choose the BERT architecture and pretraining model.", 
    ["bert-base-uncased", "alvaroalon2/biobert_diseases_ner"]) 
    
    if choice_model == "BERT Model 1":
        if choice_pretraining_model == "bert-base-uncased":
            weight_path = "Model_BERT_1_270/best_model_BERT1 (1).pt"
        elif choice_pretraining_model == "alvaroalon2/biobert_diseases_ner":
            weight_path = "Model_BERT_1_270/best_model_BERT1 (1).pt"

    elif choice_model == "BERT Model 2": 
        if choice_pretraining_model == "bert-base-uncased":
            weight_path = "Model_BERT_2_Nomal/best_model_BERT2_bert-base-uncased.pt"
        elif choice_pretraining_model == "alvaroalon2/biobert_diseases_ner":
            weight_path = "Model_BERT_2/best_model_BERT2.pt"

    tokenizer = BertTokenizer.from_pretrained(choice_pretraining_model) 


    st.subheader("Step 2: Input the sentence.", divider='rainbow') 

    col1, col2 = st.columns(2)  # Using beta_columns to create two columns

    with col1:
        e1 = st.text_input("Enter Entity e1:")

    with col2:
        e2 = st.text_input("Enter Entity e2:")
    
    if choice_model == "BERT Model 1":
        full_sentence = st.text_input("Enter Full Sentence:") 
        input_text = f"{e1} [SEP] {e2} [SEP] {full_sentence}"
        test_dataset = MyDataset( e1 = e1, e2 = e2, sentence = full_sentence , tokenizer = tokenizer, max_len = 270)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True )
        print(test_dataset["text"])
    elif choice_model == "BERT Model 2":
        test_dataset = MyDataset( e1 = e1, e2 = e2, sentence = None , tokenizer = tokenizer, max_len = 30)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True )
        print(test_dataset["text"])

    

    
    st.subheader("Step 3: Output the result.", divider='rainbow')

    if st.button("Run"):
        bert_model_name = choice_pretraining_model
        print(bert_model_name)
        print(weight_path)

        model_predict = load_bert_model(weight_path, bert_model_name)

        for data in test_loader:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            logits = model_predict(input_ids, attention_mask)

            # Get predicted labels and probabilities
            predicted_labels = torch.argmax(logits, dim=1).tolist()[0]
            predicted_probs = torch.nn.functional.softmax(logits, dim=1)[0].tolist()
            print(predicted_labels)

            # Check if the prediction is false (label 5)
            if predicted_labels == 0 :
                # Penalize false prediction by reducing probabilities
                predicted_probs_2 = torch.tensor(predicted_probs)  # Convert list to tensor
                second_best_index = sorted(range(len(predicted_probs_2)), key=lambda i: -predicted_probs_2[i])[1]
                penaty = 0.21*predicted_probs[0]  
                predicted_probs[0] = predicted_probs[0] - penaty
                predicted_probs[second_best_index] = predicted_probs[second_best_index] + penaty
                

            st.success(f"Predicted Relation: {labels[predicted_labels]}")

            # Filter out class 5
            filtered_probs = [prob for i, prob in enumerate(predicted_probs) if i != 5]

            # Create a list of labels excluding class 5
            filtered_labels = [label for i, label in enumerate(labels) if i != 5]

            # Plot the distribution using seaborn
            plt.figure(figsize=(8, 6))
            sns.barplot(x=filtered_labels, y=filtered_probs, palette='viridis')
            plt.xlabel('Labels')
            plt.ylabel('Probability')
            plt.title('Predicted Label Distribution')

            # Display percentages on top of each bar
            for i, prob in enumerate(filtered_probs):
                plt.text(i, prob + 0.01, f'{prob * 100:.2f}%', ha='center')

            st.pyplot(plt)

            st.balloons()

    # if st.button("Run"):
    #     bert_model_name = choice_pretraining_model
    #     print(bert_model_name)
    #     print(weight_path)

    #     model_predict = load_bert_model(weight_path, bert_model_name)

    #     for data in test_loader:
    #         input_ids = data["input_ids"]
    #         attention_mask = data["attention_mask"]
    #         logits = model_predict(input_ids, attention_mask)

    #         # Get predicted labels and probabilities
    #         predicted_labels = torch.argmax(logits, dim=1).tolist()[0]
    #         predicted_probs = torch.nn.functional.softmax(logits, dim=1)[0].tolist()

    #         st.success(f"Predicted Relation: {labels[predicted_labels]}")

    #         # Filter out class 5
    #         filtered_probs = [prob for i, prob in enumerate(predicted_probs) if i != 5]

    #         # Create a list of labels excluding class 5
    #         filtered_labels = [label for i, label in enumerate(labels) if i != 5]

    #         # Plot the distribution using seaborn
    #         plt.figure(figsize=(8, 6))
    #         sns.barplot(x=filtered_labels, y=filtered_probs, palette='viridis')
    #         plt.xlabel('Labels')
    #         plt.ylabel('Probability')
    #         plt.title('Predicted Label Distribution')

    #         # Display percentages on top of each bar
    #         for i, prob in enumerate(filtered_probs):
    #             plt.text(i, prob + 0.01, f'{prob * 100:.2f}%', ha='center')

    #         st.pyplot(plt)

    #         st.balloons()




    # if st.button("Run"):
    #     bert_model_name = choice_pretraining_model
    #     print(bert_model_name)
    #     print(weight_path)

    #     model_predict = load_bert_model(weight_path, bert_model_name)

    #     for data in test_loader:
    #         input_ids = data["input_ids"]
    #         attention_mask = data["attention_mask"]
    #         logits = model_predict(input_ids, attention_mask)

    #         # Get predicted labels
    #         predicted_labels = torch.argmax(logits, dim=1).tolist()[0]
    #         st.success(f"Predicted Relation: {labels[predicted_labels]}")

    #         probs = torch.nn.functional.softmax(logits, dim=1)[0].tolist()
    #         break

    #     # Filter out class 5
    #     filtered_probs = [prob for i, prob in enumerate(probs) if i != 5]

    #     # Create a list of labels excluding class 5
    #     filtered_labels = labels

    #     # Plot the distribution using seaborn
    #     plt.figure(figsize=(8, 6))
    #     sns.barplot(x=filtered_labels, y=filtered_probs, palette='viridis')
    #     plt.xlabel('Labels')
    #     plt.ylabel('Probability')
    #     plt.title('Predicted Label Distribution')

    #     # Display percentages on top of each bar
    #     for i, prob in enumerate(filtered_probs):
    #         plt.text(i, prob + 0.01, f'{prob * 100:.2f}%', ha='center')

    #     st.pyplot(plt)

    #     st.balloons()
