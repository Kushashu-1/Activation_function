import streamlit as st
from MyPackage import functions
from MyPackage import visuals

st.sidebar.title(" Menu ")
page = st.sidebar.radio("", ('Home', 'About'))

# Home page
def home():
    st.title("Activation Function Visualization")
    selected_option = st.selectbox('Select Activation Function', ('Exponential Linear Unit', 'Linear function', 'Binary function',
                                   'Softmax function', 'Leaky Relu function', 'Tanh function', 'Sigmoid function', 'Relu function', 'Hard Tanh', 'Hard Sigmoid', 'Swish function', 'Silu function'))
    
    b = '0'
    if selected_option == 'Swish function':
        b = st.selectbox('Select beta Value' , ('1','2','3'))
    
    x_from = st.sidebar.slider("Negative Baundary", 0,50,10)
    x_to = st.sidebar.slider("Positive Baundary", 0, 50,10)
    x = functions.take_range(-1*x_from,x_to)
    y = functions.generate_y(selected_option ,x,b)
    if st.button("Plot Graph"):
        fig = visuals.draw_graph(x,y,selected_option)
        st.pyplot(fig)


# About Page
def about():
    st.title("About")
    st.write("It is Basic Web App for the Visualization of Activation Function in Neural NetWork")
    st.subheader("Created By : ")
    st.write("Ashutosh Singh Kushwaha")
    st.write("Contact at : https://www.linkedin.com/in/kushashu")

# Menu
if page == 'Home':
    home()
elif page == 'About':
    about()
