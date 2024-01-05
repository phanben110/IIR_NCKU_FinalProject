import os
import streamlit as st 

import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from torchvision.models import resnet18
import torch.nn as nn
import numpy as np 
import cv2

text_list = []
import re

def remove_parentheses(file_name):
    # Sử dụng regular expression để loại bỏ các ngoặc đơn và nội dung bên trong chúng
    return re.sub(r'\([^)]*\)', '', file_name)

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

# In danh sách text_list và image_paths
# print("text_list =", text_list)
# print("image_paths =", image_paths)


# 定义Clip模型
class ClipModel(nn.Module):
    def __init__(self, text_encoder_name='bert-base-uncased'):
        super(ClipModel, self).__init__()

        # Load pre-trained BERT
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        self.tokenizer = BertTokenizer.from_pretrained(text_encoder_name)

        # Freeze BERT parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Load pre-trained ResNet18
        self.image_encoder = resnet18(pretrained=True)
        # Remove the fully connected layer of ResNet18
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])

        # Additional layer for mapping ResNet18 output to BERT output dimensions
        self.mapping_layer = nn.Linear(512, self.text_encoder.config.hidden_size)

    def forward(self, text, image):
        # Text encoding with BERT
        text_inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        text_outputs = self.text_encoder(**text_inputs, return_dict=True)
        text_pooler_output = text_outputs.pooler_output  # Using pooler output

        # Image encoding with ResNet18
        image_outputs = self.image_encoder(image)
        image_outputs = image_outputs.view(image_outputs.size(0), -1)  # Flatten the output

        # Mapping ResNet18 output to BERT output dimensions
        mapped_outputs = self.mapping_layer(image_outputs)

        return text_pooler_output, mapped_outputs

def inference(image, clip_model):
    # 读取图像
    # image = Image.open(image_path).convert('RGB')

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image)

    # 将图像移至GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)

    # 将模型移至GPU（如果可用）
    clip_model = clip_model.to(device)

    # 使用模型进行前向传播
    with torch.no_grad():
        _, image_output = clip_model("", image.unsqueeze(0))

    # 计算输入图像与每个训练文本之间的相似度
    similarities = []
    text_list.sort()

    for text in text_list:
        text_inputs = clip_model.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        text_outputs = clip_model.text_encoder(**text_inputs, return_dict=True)
        text_pooler_output = text_outputs.pooler_output

        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(image_output.squeeze(dim=1), text_pooler_output.squeeze(dim=1))
        similarities.append(similarity.item())

    # 找到相似度最大的文本
    max_similarity_index = similarities.index(max(similarities))
    most_similar_text = text_list[max_similarity_index]

    # print(text_list)
    # print(similarities)
    # print(most_similar_text)
    return most_similar_text

def clip():
    st.image("image/clip.png") 
    st.subheader("1. CLIP Architecture", divider='rainbow')
    st.image("image/CLIP_Architecture.png")    
    st.subheader("2. Inference", divider='rainbow') 

    st.sidebar.header("Load Image For Inference")
    img_infer = np.array([])

    uploaded_file = st.sidebar.file_uploader("Load Image inference", type=["png","jpg"], accept_multiple_files=False)
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        img_infer = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img_infer is not None:
            img_infer = cv2.cvtColor(img_infer, cv2.COLOR_BGR2RGB)
            st.sidebar.image(img_infer, caption="Image for inference", use_column_width=True)

        if st.button("Runing Inference"):
            if img_infer is not None:
                im = Image.fromarray(img_infer)
                trained_model_path = 'Model_CLIP/clip_model_epoch99.pth'  # 替换为你的模型权重路径
                clip_model = ClipModel()
                clip_model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))
                clip_model.eval()
                result_text = inference(im, clip_model)
                st.success(f"The most similar text for the given image is: {result_text}", icon="✅" )

            else:
                st.sidebar.error(f"Failed to read image: {uploaded_file.name}")