from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from torchvision.models import resnet18
import torch.nn as nn
import numpy as np 
import cv2
import streamlit as st  
import torch
from tqdm import tqdm
import os
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from src.p5_clip import *
from src.p4_inference import * 
warnings.filterwarnings("ignore")

device = "cpu"
labels = ["false", "effect", "mechanism", "advise", "int"] 

text_list = []
image_paths = []

with open('ouput_drug.txt', 'r') as file:
    for line in file:
        # Lấy tên file từ đường dẫn và loại bỏ ngoặc đơn
        file_name = remove_parentheses(line.strip().split("/")[-1].split(".")[0])
        
        # Thêm tên file vào danh sách text_list
        text_list.append(file_name)
        
        # Thêm đường dẫn vào danh sách image_paths
        image_paths.append(line.strip())

def extract_relation(): 
    st.image("image/Extract Relation.png") 
    st.subheader("1. Extract Relation with CLIP Architecture", divider='rainbow')
    st.image("image/CLIP_Architecture.png")    

    uploaded_file_1 = st.sidebar.file_uploader("Load Image Left", type=["png","jpg"], accept_multiple_files=False)
    uploaded_file_2 = st.sidebar.file_uploader("Load Image Right", type=["png","jpg"], accept_multiple_files=False)
    result_text_1 = None 
    result_text_2 = None 
    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        if st.button("Runing ..."): 
            st.subheader("2. Inference CLIP Model", divider='rainbow') 
            trained_model_path = 'Model_CLIP/clip_model_epoch99.pth'  # 替换为你的模型权重路径
            clip_model = ClipModel()
            clip_model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))
            clip_model.eval()
            col1, col2 = st.columns(2)

            with col1: 

                if uploaded_file_1 is not None:
                    image_data_1 = uploaded_file_1.read()
                    image_array_1 = np.frombuffer(image_data_1, np.uint8)
                    img_infer_1 = cv2.imdecode(image_array_1, cv2.IMREAD_COLOR)

                    if img_infer_1 is not None:
                        img_infer_1 = cv2.cvtColor(img_infer_1, cv2.COLOR_BGR2RGB)
                        st.image(img_infer_1, caption="Image Left", use_column_width=True, width=300)
                        im1 = Image.fromarray(img_infer_1)
                        result_text_1 = inference_clip(im1, clip_model)
                        st.success(f"{result_text_1}", icon="✅" )
            with col2:

                if uploaded_file_2 is not None:
                    image_data_2 = uploaded_file_2.read()
                    image_array_2 = np.frombuffer(image_data_2, np.uint8)
                    img_infer_2 = cv2.imdecode(image_array_2, cv2.IMREAD_COLOR)

                    if img_infer_2 is not None:
                        img_infer_2 = cv2.cvtColor(img_infer_2, cv2.COLOR_BGR2RGB)
                        st.image(img_infer_2, caption="Image Right", use_column_width=True, width=300)
                        im2 = Image.fromarray(img_infer_2)
                        result_text_2 = inference_clip(im2, clip_model)
                        st.success(f"{result_text_2}", icon="✅" )

            st.subheader("3. Extract Relation", divider='rainbow') 

            if result_text_1 is not None and result_text_2 is not None:
                choice_pretraining_model = "alvaroalon2/biobert_diseases_ner"
                weight_path = "Model_BERT_2/best_model_BERT2.pt"
                tokenizer = BertTokenizer.from_pretrained(choice_pretraining_model) 
                bert_model_name = choice_pretraining_model
                model_predict = load_bert_model(weight_path, bert_model_name) 
                test_dataset = MyDataset( e1 = result_text_1, e2 = result_text_2, sentence = None , tokenizer = tokenizer, max_len = 30)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True )
                print(test_dataset["text"])

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

                st.success(f"Relation: {result_text_1} {result_text_2}: {labels[predicted_labels]}", icon="✅" )
            else:
                st.error(f"Failed to read image: {uploaded_file_1.name} or {uploaded_file_2.name}")
    

