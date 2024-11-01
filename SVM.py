import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

st.sidebar.markdown("# Support Vector Machine")
X,Y=make_blobs(n_samples=100,centers=2,random_state=6)

def graph(svm):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Paired)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY,XX=np.meshgrid(yy,xx)
    xy = np.vstack([XX.ravel(),YY.ravel()]).T
    Z = svm.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], aplha=0.5, linestyles=['--','-','--'])
    ax.scatter(svm.support_vectors_[:,0],svm.support_vectors_[:,1],s=100,linewidth=1,facecolors='none',edgecolors='k')
    st.pyplot(plt.gcf())
    plt.clf()

C = st.sidebar.number_input("Enter C",min_value=0.0,value=1.0)
kernel = st.sidebar.selectbox(
    'Enter Kernel',
    ('rbf','linear','ploy','sigmoid','precomputed')
)
gamma = st.sidebar.selectbox(
    'Enter Gamma',
    ('scale','auto')
)
coef0 = st.sidebar.number_input("Enter coef0",min_value=0.0,value=0.0)
shrinking = st.sidebar.selectbox(
    'Enter Shrinking',
    (True,False)
)
probability = st.sidebar.selectbox(
    'Enter Probability',
    (False,True)
)
verbose = st.sidebar.selectbox(
    'Select Verbose',
    (False,True)
)
max_iter = st.sidebar.number_input("Enter Max_iter",value=-1)
decision_function_shape = st.sidebar.selectbox(
    ('Select Decision Function Shape'),
    ('ovr','ovo')
)

if st.sidebar.button("Run Algorithm"):
    svm = SVC(C=C,kernel=kernel,gamma=gamma,coef0=coef0,shrinking=shrinking,probability=probability,verbose=verbose,max_iter=max_iter,decision_function_shape=decision_function_shape)
    svm.fit(X,Y)

    st.subheader("Parameters(with values)")
    st.write("C: ",C)
    st.write("Kernel: ",kernel)
    st.write("Gamma: ",gamma)
    st.write("Coef0: ",coef0)
    st.write("Shrinking: ",shrinking)
    st.write("Probability: ",probability)
    st.write("Verbose: ",verbose)
    st.write("Max_iter: ",max_iter)
    st.write("Decision Function Shape: ",decision_function_shape)
    graph(svm)