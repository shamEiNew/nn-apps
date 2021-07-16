import mpneuron as mp
import percepalgo as perceptron
import streamlit as st



st.set_page_config(layout="wide")


apps = {'Perceptron':perceptron.main_percep, 'MCP-Neuron':mp.main}
app = st.sidebar.radio('Menu',options=list(apps.keys()), help='Simple apps for display')

if app:
    st.title(app)
    apps[app]()