import streamlit as st
from source.perceptron import *

#Path to sample data of size 100*3
path = "source\\data\\data.csv"

#Getting features of size 100*2 and 100*1
X, y = prepare(path)

def upper_columns():

    cols = st.beta_columns(3)
    with cols[0]:
        st.header('Early Perceptron on Machine')
        st.image('./img/Mark_I_perceptron.jpeg', caption="Mark I Machine")

    with cols[1]:
        st.header('Algorithm')
        st.write("The perceptron is an algorithm which is linear binary classifier")
        st.image('./img/perceptron.png')
        st.write(""" It takes the inputs/features and
                outputs boolean value passing through some output function, the following
                example uses step function as output.
                """)

        st.write("""
        The algorithm converges as far as the
        as inherently it keeps changing angle which forms a bounded sequence. The proof can be 
        found [here](https://shamsundarpanchal.wordpress.com/).""")

    with cols[2]:
        st.header('Description')
        st.write(
            """
            Let $X, W, f$ denote the features of size $(m, n)$, weights of size $(n, 1)$, output function respectively.
            For each row i in X, we have
            $$ 
            y=1 \\text{ if } X[i]\cdot W>0
            $$

            $$ 
            y=0 \\text{ if } X[i]\cdot W<0
            $$

            Weight Updates are made as follows:
            If negative labelled point is misclassified that it is predicted positive.

            $$ 
            W =  W - \\alpha \cdot X[i]
            $$

            If a positive labelled point is misclassified that it is predicted negative.
            $$
            W = W + \\alpha \cdot X[i]
            $$
            Where alpha denotes the learning rate.
            """
        )
    



    
#Function for app excecution
def main_percep():
    upper_columns()
    columns = st.beta_columns(3)

    with columns[0]:
        st.header('INPUTS')
        #User inputs as x, y, label
        user_X1 = st.text_input('Feature:1',help='x-variable') #x1
        user_X2 = st.text_input('Feature:2',help='y-variable') #x2
        user_y = st.text_input('Labels',help='label 1 or 0') #y
        epochs = st.text_input('Number of Epochs', value='25')

        if (user_X1) and (user_X2) and (user_y):

            #input variable
            x1= list(map(float, user_X1.split(',')))
            x2= list(map(float, user_X2.split(',')))
            epochs = int(epochs)
            #target variable
            y1= np.array(list(map(float, user_y.split(","))))

            #Creating array of from user input
            X1 = zip(x1, x2)
            X1 = np.array(list(map(list, X1)))
            try:
                #Display user data
                st.write(pd.DataFrame({'x':x1, 'y':x2, 'label':y1}))
            except:
                st.write('Enter arrays of equal length')

        else:
            st.write('Sample Data')
            st.write(pd.read_csv(path))


    with columns[1]:
        
        #Plot points of data

        st.header('Scatter Plot')
        if (user_X1) and (user_X2) and (user_y):
            try:
                fig = plot_points(X1, y1)
                st.write(fig)
            except:
                st.write('Enter valid values and arrays of equal length.')
        else:
            fig = plot_points(X, y)
            st.write(fig)

       

    with columns[2]:
        #training the perceptron on boundary
        
        st.header('Boundary Line')
        with st.spinner(text='Evaluating lines...'):
            if (user_X1) and (user_X2) and (user_y):
                try:
                    boundary_lines = trainPerceptronAlgorithm(X1, y1, num_epochs=epochs)
                    fig_lines = fig
                except:
                    st.write('Enter valid values and arrays of equal length.')

            else:
                boundary_lines = trainPerceptronAlgorithm(X, y)
                fig_lines = fig

            
            try:
                #getting slope and intercept values for each line
                for m, b in boundary_lines[:len(boundary_lines)-1]:
                    bd_lines = display(fig_lines, m ,b)

                #plotting the final boundary. It is the last slope value in bd_lines.
                if (user_X1) and (user_X2) and (user_y):
                    bd_lines = display(bd_lines, boundary_lines[-1][0], boundary_lines[-1][1], 'black', label='Boundary Line')
                else:
                    bd_lines = display(bd_lines, boundary_lines[-1][0], boundary_lines[-1][1], 'black',x_min = [-0.05, 1.05],y_min=[-0.05, 1.05], label='Boundary Line')
                st.write(bd_lines)
                st.write('The algorithm converges only when the datapoints are linearly separable.')
            except:
                pass
