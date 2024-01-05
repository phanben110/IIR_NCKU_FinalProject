import streamlit as st

import os
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_xml_files(input_folder):
    # Initialize lists to store entity data
    entity_data = {'ID': [], 'ID entity': [], 'Offset': [], 'Text': [], 'Type': [], 'Full Sentence': []}

    # Initialize lists to store pair data
    pair_data = {'ID': [], 'ID pair': [], 'ID e1': [], 'ID e2': [],
                 'entity e1': [], 'entity e2': [], 'ddi': [], 'pair type': [], 'Full Sentence': []}

    # Iterate through each XML file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".xml"):
            file_path = os.path.join(input_folder, filename)

            # Parse the XML file
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Iterate through each sentence in the XML
            for sentence in root.findall('.//sentence'):
                sentence_id = sentence.attrib['id']
                sentence_text = sentence.attrib['text']

                # Iterate through each entity in the sentence
                for entity in sentence.findall('.//entity'):
                    entity_id = entity.attrib['id']
                    offset = entity.attrib['charOffset']
                    text = entity.attrib['text']
                    entity_type = entity.attrib['type']

                    # Append entity data to lists
                    entity_data['ID'].append(sentence_id)
                    entity_data['ID entity'].append(entity_id)
                    entity_data['Offset'].append(offset)
                    entity_data['Text'].append(text)
                    entity_data['Type'].append(entity_type)
                    entity_data['Full Sentence'].append(str(sentence_text).rstrip()) 

                # Iterate through each pair in the sentence
                for pair in sentence.findall('.//pair'):
                    pair_id = pair.attrib['id']
                    e1_id = pair.attrib['e1']
                    e2_id = pair.attrib['e2']
                    ddi = pair.attrib['ddi']

                    # Determine the type based on the 'ddi' attribute
                    pair_type = "false" if ddi == 'false' else pair.attrib.get('type', None)

                    # Find the corresponding entities
                    e1_entity = entity_data['Text'][entity_data['ID entity'].index(e1_id)]
                    e2_entity = entity_data['Text'][entity_data['ID entity'].index(e2_id)]

                    # Append pair data to lists
                    pair_data['ID'].append(sentence_id)
                    pair_data['ID pair'].append(pair_id)
                    pair_data['ID e1'].append(e1_id)
                    pair_data['ID e2'].append(e2_id)
                    pair_data['entity e1'].append(e1_entity)
                    pair_data['entity e2'].append(e2_entity)
                    pair_data['ddi'].append(ddi)
                    pair_data['pair type'].append(pair_type)
                    pair_data['Full Sentence'].append(str(sentence_text).rstrip())

    # Create DataFrames from the data
    entity_df = pd.DataFrame(entity_data)
    pair_df = pd.DataFrame(pair_data)

    return entity_df, pair_df

def processing():
    st.image("image/1_processing.png")
    # Step 1: Choose dataset
    st.subheader("Step 1: Data Processing", divider='rainbow')
    status_step1 = False
    dataset_choice = st.selectbox("Choose Dataset", 
    ["Train Dataset", "Devel Dataset", "Test Dataset"])
    path_dataset = ""
    if dataset_choice == "Train Dataset":
        path_dataset = "SemEval-2013-task-9/data/Train" 
    elif dataset_choice == "Devel Dataset": 
        path_dataset = "SemEval-2013-task-9/data/Devel" 
    elif dataset_choice == "Test Dataset":
        path_dataset = "SemEval-2013-task-9/data/Test-DDI" 
    # Step 2: 
    show_rawdata = st.checkbox("Show the Raw Data", value=False)
    xml_file_path = "SemEval-2013-task-9/data/Train/163470.xml" 
    if show_rawdata: 
        st.markdown(f'<p style="text-align:center; color:red;">Raw Data</p>', unsafe_allow_html=True) 
        with open(xml_file_path, "r") as file:
            xml_content = file.read()

        st.code(xml_content, language='xml')
    show_dataframe = st.checkbox("Show the Entities and Pairs Dataframes", value=False) 
    if path_dataset != "" and show_dataframe:
        entity_df, pair_df = process_xml_files(path_dataset)
        st.markdown(f'<p style="text-align:center; color:red;">Entities Dataframe</p>', unsafe_allow_html=True)
        st.dataframe(entity_df)
        st.markdown(f'<p style="text-align:center; color:red;">Pair Dataframe</p>', unsafe_allow_html=True) 
        st.dataframe(pair_df) 
        status_step1 = True 

    st.subheader("Step 2: Exploratory Data Analysis", divider='rainbow' ) 
    show_dis_entity_type = st.checkbox("Show the Distribution of Entity Type", value=False) 
        
    if show_dis_entity_type:
        #st.markdown(f'<p style="text-align:center; color:red;">Distribution of Entity Type</p>', unsafe_allow_html=True) 
        # Plot a bar chart for the distribution of entity types in entity_df_result
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Type', data=entity_df)
        plt.title('Distribution of Entity Types')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        st.pyplot(plt) 
    show_dis_DDI_type = st.checkbox("Distribution of DDI Values in pairs dataframe", value=False) 
    if show_dis_DDI_type: 
        #st.markdown(f'<p style="text-align:center; color:red;">Distribution of DDI Values in pairs dataframe</p>', unsafe_allow_html=True) 
        plt.figure(figsize=(8, 8))
        pair_df['ddi'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of DDI Values in Pairs')
        st.pyplot(plt) 

    show_dis_pair_type = st.checkbox("Distribution of Pair Type", value=False) 
    if show_dis_pair_type:  
        #st.markdown(f'<p style="text-align:center; color:red;">Distribution of Pair Type</p>', unsafe_allow_html=True)
        plt.figure(figsize=(10, 6))
        sns.countplot(x='pair type', data=pair_df)
        plt.title('Distribution of Pair Types')
        plt.xlabel('Pair Type')
        plt.ylabel('Count')
        st.pyplot(plt)

    st.subheader("Step 3: Prepare the Dataset for training BERT Model", divider='rainbow' ) 


    show_short_dataset = st.checkbox("Extract the following information: entity e1, entity e2, pair type, Full Sentence.", value=False)
    if show_short_dataset:
        st.markdown(f'<p style="text-align:center; color:red;">Short Dataset</p>', unsafe_allow_html=True)
        # Create a new DataFrame with the columns we need
        data_frame = pair_df.loc[:, ['entity e1', 'entity e2', 'pair type',"Full Sentence"]] 
        st.dataframe(data_frame)
        st.info(f"The number of rows in DataFrame: {data_frame.shape[0]}", icon="ℹ️" ) 

    show_agument_dataset = st.checkbox("Agument Dataset", value=False)
    if show_agument_dataset: 
        st.markdown(f'<p style="text-align:center; color:red;">Agument pairs dataframe</p>', unsafe_allow_html=True)
        data_frame_2 = pair_df.loc[:, ['entity e1', 'entity e2', 'pair type',"Full Sentence"]]
        new_rows = {'entity e1': data_frame_2['entity e2'],
                    'entity e2': data_frame_2['entity e1'],
                    'pair type': data_frame_2['pair type'],
                    'Full Sentence': data_frame_2['Full Sentence']}
        new_data_frame = data_frame_2.append(pd.DataFrame(new_rows), ignore_index=True)
        st.dataframe(new_data_frame) 
        st.info(f"The number of rows in the original DataFrame: {new_data_frame.shape[0]}", icon="ℹ️")

    show_drop_duplicate_dataset = st.checkbox("Drop Duplicate Dataset", value=False) 
    if show_drop_duplicate_dataset:
        st.markdown(f'<p style="text-align:center; color:red;">Drop Duplicate Dataset</p>', unsafe_allow_html=True)
        new_data_frame.drop_duplicates(inplace=True) 
        st.dataframe(new_data_frame) 
        st.info(f"The number of rows in the new DataFrame: {new_data_frame.shape[0]}",icon="ℹ️")
    
    if st.button("Export to CSV"): 
        st.balloons()
    

