import streamlit as st
from streamlit_option_menu import option_menu
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
from src.utils import parse_xml
from src.utils import search_and_highlight
from src.utils import *
import re
from Bio import Entrez
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import PorterStemmer
from src.p1_processing import *
from src.p2_training import *
from src.p3_evaluation import *
from src.p4_inference import *
from src.p5_clip import *
from src.footer import settingFooter


# Create an option menu for the main menu in the sidebar
st.set_page_config(page_title="Final Project IR", page_icon="image/logo_csie2.png")
# st.image("image/title_search.png")
st.sidebar.image("image/logo_NCKU.jpeg", use_column_width=True)
with st.sidebar:
    selected = option_menu("Main Menu", ["1. Data Processing", "2. BERT Training", "3. Evaluation", "4. Inference", "5. CLIP"],
                           icons=["blockquote-right","cpu-fill", "bar-chart-fill", "body-text" , "clipboard-data-fill"], menu_icon="bars", default_index=0)
# Based on the selected option, you can display different content in your web application
# page for select icon https://icons.getbootstrap.com/

# settingFooter()
if selected == "1. Data Processing":
    processing()

elif selected == "2. BERT Training":
    training()

elif selected == "3. Evaluation":
    evaluation()

elif selected == "4. Inference":
    inference()

elif selected == "5. CLIP":
    clip()