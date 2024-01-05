import os
import streamlit as st  

def training():
    st.image("image/2_training.png")
    st.subheader("Step 1: BERT Architecture for Classification", divider='rainbow')
    show_bert_model_1 = st.checkbox("BERT Model 1", value=False)
    if show_bert_model_1: 
        st.markdown(f'<p style="text-align:center; color:red;">BERT Model 1</p>', unsafe_allow_html=True) 
        st.image("image/Model1.png")
    show_bert_model_2 = st.checkbox("BERT Model 2", value=False)
    if show_bert_model_2:
        st.markdown(f'<p style="text-align:center; color:red;">BERT Model 2</p>', unsafe_allow_html=True) 
        st.image("image/Model2.png")

    st.subheader("Step 2: Split Dataset", divider='rainbow') 
    show_split_dataset = st.checkbox("Split Dataset", value=False) 
    if show_split_dataset:
        st.markdown(f'<p style="text-align:center; color:red;">Split Dataset</p>', unsafe_allow_html=True) 
        st.image("image/dataset.png") 

    st.subheader("Step 3: Loss Function", divider='rainbow')
    show_loss_function = st.checkbox("Focus Loss Function", value=False) 
    if show_loss_function:
        st.image("image/Focusloss.png")
        st.info("Focal Loss is a modified cross-entropy loss that introduces two additional parameters 𝛼 and 𝛾, to handle class imbalance and emphasize hard examples during training.")

    st.subheader("Step 4: Choose Pretrained Model BERT", divider='rainbow')
    model_choice = st.selectbox("Choose Pretrained Model BERT", 
    ["bert-base-uncased", "alvaroalon2/biobert_diseases_ner"])

    st.subheader("Step 5: Setting Parameters", divider='rainbow')

    show_setting_parameters = st.checkbox("Setting Parameters", value=False) 
    if show_setting_parameters:
        st.write("1. **Learning Rate:**")
        st.write("   - I choose a learning rate of 1e-5 for fine-tuning BERT models. Adjustments may be made based on empirical performance.")

        st.write("2. **Batch Size:**")
        st.write("   - For training on a Kaggle GPU P100, I set the batch size to 64 for BERT model 1 and 512 for BERT model 2.")

        st.write("3. **Number of Epochs:**")
        st.write("   - I set the number of epochs to 100 and monitor the validation performance.")

        st.write("4. **Optimizer:**")
        st.write("   - I use the Adam optimizer, which is commonly used for fine-tuning BERT models.")

        st.write("5. **Warmup Steps:**")
        st.write("   - I implement a warmup strategy for the learning rate, gradually increasing it during the initial training steps.")

        st.write("6. **Weight Decay:**")
        st.write("   - I apply weight decay to prevent overfitting. A common value is 1e-2 or 1e-3.")



