import mpneuron as mp
import percepalgo as perceptron
import streamlit as st
from source import digitclassify as digitc



st.set_page_config(page_title="Neural Apps",layout="wide", page_icon="🧠")

apps = {'Perceptron':perceptron.main_percep, 'MCP-Neuron':mp.main, "Digit Classifier":digitc.main} 
app = st.sidebar.radio('Menu',options=list(apps.keys()), help='Simple apps for display')

if app:
    st.title(app)
    apps[app]()