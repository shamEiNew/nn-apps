import streamlit as st
from PIL import Image
from source.BooleanFunctions import BooleanFunctions, BoolPlots




def main():
    st.write("""
    In 1943 Warren S. McCulloch, a neuroscientist, and Walter Pitts, a logician,
    published "A logical calculus of the ideas immanent in nervous activity" in the Bulletin of
    Mathematical Biophysics 5:115-133.
    In this paper McCulloch and Pitts tried to understand how the brain
    could produce highly complex patterns by using many basic cells
    that are connected together. 
    These basic brain cells are called neurons, and McCulloch and Pitts gave a highly
    provided a simplified model of a neuron in their paper. The McCulloch and Pitts model of a neuron,
    which we will call an MCP neuron for short,
    has made an important contribution to the development of artificial neural networks
    -- which model key features of biological neurons.
    This extract is take from [here](https://mind.ilstu.edu/curriculum/mcp_neurons/index.html).
    \n
    This simple idea lead to the development of perceptron and finally we today have 
    neural networks.
    """)


    column =  st.columns(3)
    plots = BoolPlots()

    with column[0] as c1:
        img = Image.open('./img/mpneuron.png')
        st.header('McCulloch-Pitts Neuron')
        st.image(img, use_column_width=True)
        st.write(
            """
            MP Neuron is the earliest form of artificial neuron where
            g is the aggregate function and f depending upon the threshold
            outputs the value.

            $$
            g(x)=\sum_i^{n}x_i
            $$
            $$
            if \\ g(x)\geq \\theta
            \\ 
            y=f(g(x))=1
            $$
            $$
            if \\ g(x)< \\theta
            \\ 
            y=f(g(x))=0
            $$

            """
        )
    with column[1] as c2:
        st.header('Boolean Functions')
        bool_func = st.selectbox('Functions', options=['AND','OR','NAND', 'NOR'])
        input_val = st.text_input('INPUT', help='input 1 or 0 sepeated by ","')
        if input_val:
            x = list(map(int, input_val.strip().split(',')))
            func = BooleanFunctions(x)
        

        try:
            if bool_func =='AND':
                val = func.boolAND()
                st.write(val[0])

            elif bool_func =='OR':
                val = func.boolOR()
                st.write(val[0])

            elif bool_func =='NAND':
                val = func.notAND()
                st.write(val[0])

            elif bool_func =='NOR':
                val = func.notOR()
                st.write(val[0])
        except:
            st.markdown('Please provide input values separted by ","')
        



        
    with column[2]:
        st.header('GRAPH')
        try:
            st.write(plots.linePlot(val[1]))
        except:
            st.markdown('Please provide input values')