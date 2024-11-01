import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

st.sidebar.markdown("# DBSCAN Algorithm")
Dataset = st.sidebar.selectbox(
    "Select Dataset",
    ('Select Dataset','Iris','Blobs','Moon',"Circle")
)

eps = st.sidebar.number_input("Enter Epsilon",min_value=0.0,value=0.5)
min_samples = st.sidebar.number_input("Enter Min Samples",min_value=1,value=5)
metric = st.sidebar.selectbox(
    'Select Metrics',
    ('euclidean','str','callable')
)
metric_params = st.sidebar.selectbox(
    'Select Metric Params',
    (None,'dic')
)
algorithm = st.sidebar.selectbox(
    'Select Algorithm',
    ('auto','ball_tree','kd_tree','brute')
)
leaf_size = st.sidebar.number_input("Enter Leaf Size",min_value=1,format="%d",value=30)
p = st.sidebar.number_input("Enter P",min_value=1,format="%d",value=2)
n_jobs = st.sidebar.selectbox(
    'Enter n_jobs',
    (1,-1)
)

if Dataset == 'Iris':
    st.subheader("Parameters(with values")
    st.write("eps: ", eps)
    st.write("min_samples: ", min_samples)
    st.write("metric: ", metric)
    st.write("metric_params: ", metric_params)
    st.write("algorithm: ", algorithm)
    st.write("leaf_size: ", leaf_size)
    st.write("p: ", p)
    st.write("n_jobs: ", n_jobs)
    iris = load_iris()
    x = iris.data
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params, algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs)
    labels = dbscan.fit_predict(x)

    fig1,ax1=plt.subplots()
    ax1.scatter(x[:,0],x[:,1],c=labels,s=30,cmap=plt.cm.Paired)

    for i in range(len(labels)):
        if labels[i] ==-1:
            ax1.scatter(x[i,0],x[i,1],s=100,color='k',label='Noise',marker='x')

    fig2, ax2 = plt.subplots()
    ax2.scatter(x[:, 2], x[:, 3], c=labels, s=30, cmap=plt.cm.Paired)

    for i in range(len(labels)):
        if labels[i] == -1:
            ax2.scatter(x[i, 0], x[i, 1], s=100, color='k', label='Noise', marker='x')

    if st.sidebar.button("Run Algorithm"):
        st.subheader("Sepal(Length and Width)")
        st.pyplot(fig1)
        st.subheader("Petal(Length and Width)")
        st.pyplot(fig2)
elif Dataset == 'Blobs':
    st.subheader("Parameters(with values")
    st.write("eps: ",eps)
    st.write("min_samples: ",min_samples)
    st.write("metric: ",metric)
    st.write("metric_params: ",metric_params)
    st.write("algorithm: ",algorithm)
    st.write("leaf_size: ",leaf_size)
    st.write("p: ",p)
    st.write("n_jobs: ",n_jobs)
    X,Y=make_blobs(n_samples=300,centers=3,cluster_std=0.60,random_state=42)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params, algorithm=algorithm,
                    leaf_size=leaf_size, p=p, n_jobs=n_jobs)
    labels = dbscan.fit_predict(X)

    plt.figure(figsize=(8,6))

    unique_labels = set(labels)
    colors = plt.cm.Paired(np.linspace(0, 1, len(unique_labels)))


    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'k'
        plt.scatter(X[labels == label, 0], X[labels == label ,1], s=30, color=color, label=f'Cluster {label}')

    if st.sidebar.button("Run Algorithm"):
        st.pyplot(plt.gcf())
        plt.clf()
elif Dataset == 'Moon':
    st.subheader("Parameters(with values")
    st.write("eps: ", eps)
    st.write("min_samples: ", min_samples)
    st.write("metric: ", metric)
    st.write("metric_params: ", metric_params)
    st.write("algorithm: ", algorithm)
    st.write("leaf_size: ", leaf_size)
    st.write("p: ", p)
    st.write("n_jobs: ", n_jobs)
    X,Y=make_moons(n_samples=300,noise=0.05)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params, algorithm=algorithm,
                    leaf_size=leaf_size, p=p, n_jobs=n_jobs)
    labels = dbscan.fit_predict(X)

    fig,ax = plt.subplots()
    ax.scatter(X[:,0],X[:,1],c=labels,s=30,cmap=plt.cm.Paired)

    if st.sidebar.button("Run Algorithm"):
        st.pyplot(fig)
        plt.clf()
elif Dataset == 'Circle':
    st.subheader("Parameters(with values")
    st.write("eps: ", eps)
    st.write("min_samples: ", min_samples)
    st.write("metric: ", metric)
    st.write("metric_params: ", metric_params)
    st.write("algorithm: ", algorithm)
    st.write("leaf_size: ", leaf_size)
    st.write("p: ", p)
    st.write("n_jobs: ", n_jobs)
    X,Y=make_circles(n_samples=300,noise=0.1,factor=0.5,random_state=42)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params, algorithm=algorithm,
                    leaf_size=leaf_size, p=p, n_jobs=n_jobs)
    labels = dbscan.fit_predict(X)

    plt.figure(figsize=(8,6))

    unique_labels = set(labels)
    colors = plt.cm.Paired(np.linspace(0, 1, len(unique_labels)))


    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'k'
        plt.scatter(X[labels == label, 0], X[labels == label ,1], s=30, color=color, label=f'Cluster {label}')

    if st.sidebar.button("Run Algorithm"):
        st.pyplot(plt.gcf())
        plt.clf()