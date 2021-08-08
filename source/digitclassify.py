from torch import nn
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas


def trained_model(img):
    model = nn.Sequential(nn.Linear(784, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10),
                        nn.LogSoftmax(dim=1))

    model.load_state_dict(torch.load("../nn-apps/source/models/digit_classifier_g.pt"))
    model.eval()
    
    with torch.no_grad():
        logits = st.session_state['model'].forward(img.view(1, 784).float())

        # #Since we used log softmax will use softmax to reverse it and get probabilites
    ps = torch.exp(logits)
    index_n = list(ps.numpy()[0])
    predicted_digit = index_n.index(max(index_n))
    
    return ps, predicted_digit


def plot_input(img):
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(6, 6))

    ax.imshow(img.squeeze(),cmap="Greys_r")
    

    return fig

def plot_prob(img):
    fig, ax0 = plt.subplots(nrows=1, ncols=1,figsize=(6, 6))

    ps, predicted_digit = trained_model(img)
    sns.barplot(y=list(range(10)), x=ps.numpy()[0], ax=ax0, orient='h')

    return fig, predicted_digit


def canvas(columns):
    SIZE = 192
    
    with columns[0]:
        
        st.header("USER INPUT")
        mode = st.checkbox("Draw", True)
        canvas_result = st_canvas(
            fill_color='#000000',
            stroke_width=20,
            stroke_color='#FFFFFF',
            background_color='#000000',
            width=SIZE,
            height=SIZE,
            drawing_mode="freedraw" if mode else "transform",
            key='canvas')
    
    with columns[1]:
        st.header("MODEL INPUT")
        if canvas_result.image_data is not None:
            img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
            rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            st.write("Input given as tensor to the model")
            st.image(rescaled)

    
    if canvas_result.image_data is not None:
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test_x.reshape(1, 28, 28)
    
    return torch.tensor(test_x/255)

def main():
    columns = st.columns(4)
    tensor = canvas(columns)
    with columns[2]:
        st.header("IMAGE PLOT OF USER INPUT")
        st.write("User input plot")
        st.write(plot_input(tensor))

    with columns[3]:
        st.header("PROBABILTY PLOT")
        st.write("Probabilty of each digit calculated by the model")
        fig, prediction = plot_prob(tensor)
        st.write(f"Predicted Digit by Classifier: {prediction}")
        st.write(fig)
    colsu = st.columns(2)


    with colsu[0]:
        st.write(
            """
            A forward network which takes input of batch of 64 images with image as row vector,\n that is 64*784 tensor\n
            is built with torch.nn module.

            -- Input  Layer 0: Inputs to the network -- Input\n
            -- Hidden Layer 1: Input Layer 0, 128 neurons each with 784 weights, Relu activation\n
            -- Hidden Layer 2: Input Layer 1, 64 neurons each with 128 weights, Relu activation \n
            -- Output Layer 3: Input Layer 1, 10 neurons each with 64 weights, output: Logsoftmax\n
            """
        )
