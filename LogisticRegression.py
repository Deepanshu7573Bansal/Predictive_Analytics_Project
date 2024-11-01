# All library import
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset graph(according the type of graph)
def load_initial_graph(dataset,ax,centers):
    if dataset == "Binary":
        x, y = make_blobs(n_features=2,centers=centers,random_state=6)
        ax.scatter(x.T[0], x.T[1], c=y, cmap='rainbow')
        return x,y
    elif dataset == "Multiclass":
        x, y = make_blobs(n_features=2,centers=centers,random_state=6)
        ax.scatter(x.T[0], x.T[1], c=y, cmap='rainbow')
        return x, y

# draw_meshgrid function
def draw_meshgrid():
    a = np.arange(start=x[:,0].min()-1, stop=x[:,0].max()+1, step=0.01)
    b = np.arange(start=x[:,1].min()-1, stop=x[:,1].max()+1, step=0.01)

    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array

# Sidebar heading
st.sidebar.markdown("# Logistic Regression")

# Sidebar options(selectbox, number_input)
# Dataset type
Dataset=st.sidebar.selectbox(
    'Select Dataset',
    ('Binary','Multiclass')
)

# Penalty(l1, l2, elasticnet)
Penalty=st.sidebar.selectbox(
    "Select Penalty",
    ('l1','l2','elasticnet','none')
)

# Value of c
C=st.sidebar.number_input("Enter C",value=1.0)

# Solver(lbfgs, liblinear, newton-cg, newton-cholesky, sag, saga)
Solver=st.sidebar.selectbox(
    "Select Solver",
    ('lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga')
)

# Number of iterations
Iteration=st.sidebar.number_input("Enter number of iteration",step=1,format="%d",min_value=0)

# Multi class(auto, ovr, multinomial)
Multi_Class=st.sidebar.selectbox(
    "Enter Multi_Class",
    ('auto','ovr','multinomial')
)

# Value of l1_ratio
L1_Ratio=st.sidebar.number_input("Enter L1 Ratio",min_value=0)

# Graph plot
fig,ax=plt.subplots()

# Condition of binary or multiclass dataset
if Dataset == "Multiclass":
    centers = st.number_input("Enter centers", min_value=2, format="%d")
    x, y = load_initial_graph(Dataset, ax, centers)
else:
    x, y = load_initial_graph(Dataset, ax,2)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
orig=st.pyplot(fig)

# Button to run algorithm
if st.sidebar.button('Run Algorithm'):
    # first clear graph(on the time of run)
    orig.empty()

    # Algorithm(parameters, train, test)
    clf = LogisticRegression(penalty=Penalty,C=C,solver=Solver,max_iter=Iteration,multi_class=Multi_Class,l1_ratio=L1_Ratio)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    XX,YY,input_array=draw_meshgrid()
    labels=clf.predict(input_array)

    # Parameters(with values
    st.subheader("Parameters(with values)")
    st.write("Penalty: ",Penalty)
    st.write("C: ", C)
    st.write("solver: ", Solver)
    st.write("max_iter: ", Iteration)
    st.write("multi_class: ", Multi_Class)
    st.write("l1_ratio: ", L1_Ratio)

    # Result as a graph(also print accuracy of logistic regression)
    ax.contourf(XX,YY,labels.reshape(XX.shape),alpha=0.5,cmap='rainbow')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    origin=st.pyplot(fig)
    st.subheader("Accuracy for Logistic Regression:"+str(round(accuracy_score(y_test,y_pred),2)))