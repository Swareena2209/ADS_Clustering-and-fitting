# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:28:53 2023

@author: sware
"""
#import all the necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import scipy.optimize as opt
from sklearn.cluster import KMeans
from scipy.stats import norm
import seaborn as sns
from scipy.optimize import curve_fit
import itertools as iter

#create function to display the graph, it includes the common features present in all graphs such as xlabel, ylabel
def display():
    plt.xlabel("Year")
    plt.ylabel("Methane emission rate(ppm)")
    plt.title("Methane emission rate")
    plt.legend(bbox_to_anchor=(1.3, 1), loc="upper right")
    plt.show()
    
#create function to plot the line graph of the dataset and assign required argument values to function plot
def line_graph(data):
    plt.plot(data[:,0], data[:,1],label="Australia",)
    plt.plot(data[:,0], data[:,2],label="Bahamas",)
    plt.plot(data[:,0], data[:,3],label="Bolivia",)
    plt.plot(data[:,0], data[:,4],label="Colombia",)
    plt.plot(data[:,0], data[:,5],label="China",)
    plt.plot(data[:,0], data[:,6],label="Spain",)
    plt.plot(data[:,0], data[:,7],label="Nigeria",)
    plt.plot(data[:,0], data[:,8],label="Poland",)
    plt.plot(data[:,0], data[:,9],label="Malta",)
    plt.plot(data[:,0], data[:,10],label="Brazil",)
    display() #call the display function to print the graph

#create function to plot the scatter plot of the dataset and assign required argument values to function scatter
def scatter_plot(df):
    plt.scatter(df[:,0], df[:,1], label ='Australia')
    plt.scatter(df[:,0], df[:,2], label ='Bahamas' )
    plt.scatter(df[:,0], df[:,3], label ='Bolivia' )
    plt.scatter(df[:,0], df[:,4], label ='Colombia')
    plt.scatter(df[:,0], df[:,5], label ='China')
    plt.scatter(df[:,0], df[:,6], label ='Spain')
    plt.scatter(df[:,0], df[:,7], label ='Nigeria')
    plt.scatter(df[:,0], df[:,8], label ='Poland')
    plt.scatter(df[:,0], df[:,9], label ='Malta')
    plt.scatter(df[:,0], df[:,10],label ='Brazil')
    display() #call the display function to print the graph
  
#create function to display pairplot using sns module
def pair():
    pdata = pd.read_csv(r"C:\Users\sware\OneDrive\Desktop\Data Science\ADS-ASSIGNMENT 3-Clustering snd fitting\clust.csv")
    sns.pairplot(pdata, hue='Country', height =2)
    plt.show()

#create function to calculate k-means clusters and for plotting the result
def kmeans_cluster():
    dclust = np.genfromtxt(r"C:\Users\sware\OneDrive\Desktop\Data Science\ADS-ASSIGNMENT 3-Clustering snd fitting\clust.csv", delimiter =',',skip_header =1)
    k_means = KMeans(n_clusters=3, max_iter=50, random_state=1) #evaluate the k-means cluster using kmeans module
    labels= k_means.fit_predict(cdata) #dataframe to store cluster labels assigned to each point using kmeans module
    centroids = k_means.cluster_centers_ #dataframe to store the centriods of each cluster using kmeans module
    print("***Lables ***\n", labels)
    print("***Centriods***\n",centroids)
    #plot the clusters and their corresponding points using different colour schemes
    filter_label0 = dclust[labels == 0]
    plt.scatter(filter_label0[:,1] , filter_label0[:,2] , color = 'green', label ="Cluster 0", marker ="o")
    filter_label1 = dclust[labels == 1]
    plt.scatter(filter_label1[:,1] , filter_label1[:,2] , color = 'purple', label="Cluster 1", marker="*")
    filter_label2 = dclust[labels == 2]
    plt.scatter(filter_label2[:,1] , filter_label2[:,2] , color = 'blue', label="Cluster 2",marker ="^")
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'black', marker ="x", label ="Centriod")
    display() #call the display function to print the graph

#create function to compare agricultural land area and methane emisson rate in Australia
def agr(medata, agdata):
    plt.plot(medata["Year"], medata["Australia"], color="black") #plot methane emission rate
    plt.title("Methane emisson rate in Australia")
    plt.xlabel("Year")
    plt.ylabel("Methane emission rate(ppm)")
    plt.show()
    plt.plot(agdata["Year"], agdata["Australia"], color="green", label="Agri", linestyle ="dashed") #plot agricultural land area
    plt.title("Agricultural land in Australia")
    plt.xlabel("Year")
    plt.ylabel("Agricultural land (%)")
    plt.show()

#create function to calculate the linear function that best fits the curve
def fct(x, a, b, c):
    return a*x**2+b*x+c #equation of a line

#function to calculate the error range
def err_ranges(x, func, param, sigma):
    lower = func(x, *param) #lower limit
    upper = lower #upper limit
    uplow = []
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
    pmix = list(iter.product(*uplow))
    for p in pmix:
            y = func(x, *p)
            lower = np.minimum(lower, y)
            upper = np.maximum(upper, y)
    return lower, upper

#create function to plot the fit and the original data 
def curve(dc,cnt):
    x,y = dc["Year"], dc[cnt] #assign the values of xaxis and yaxis
    prmet, cov = opt.curve_fit(fct, x, y) #call the curve_fit  function to determine the parameters and covariance
    dc["pop_log"] = fct(x, *prmet)
    print("Parameters are:", prmet)
    print("Covariance are:", cov)
    plt.plot(x, dc["pop_log"], label="Fit") #plot the fit
    plt.plot(x, y, label="Data") #plot the original data points
    plt.grid(True)
    plt.xlabel('Year')
    plt.ylabel('Methane emissions')
    plt.title("Methane emission rate")
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.show()
    sigma = np.sqrt(np.diag(cov)) #calculate sigma value
    print("Sigma",sigma)
    low, up = err_ranges(x, fct, prmet, sigma)
    low, up = err_ranges(2025, fct, prmet, sigma) #predcit the values of limits for the year 2050
    print("\n \n Forcasted Methane emission rate of",cnt," at 2025 is between\n", low, "and", up)
    
 #read the files in pandas and numpy formats into the dataframes   
dnump = np.genfromtxt(r"C:\Users\sware\OneDrive\Desktop\Data Science\ADS-ASSIGNMENT 3-Clustering snd fitting\meth_filter.csv", delimiter =',',skip_header =1)
dpand = pd.read_csv(r"C:\Users\sware\OneDrive\Desktop\Data Science\ADS-ASSIGNMENT 3-Clustering snd fitting\meth_filter.csv")
agri = pd.read_csv(r"C:\Users\sware\OneDrive\Desktop\Data Science\ADS-ASSIGNMENT 3-Clustering snd fitting\Agri_filtcsv.csv")

#call the pre defined functions to plot and evaluate the graphs
line_graph(dnump) #call the line function to plot the line graph
scatter_plot(dnump) #call scatter function to plot scatter plot
pair() #call the pair function to plot pairplot
kmeans_cluster() #call the kmeans function to evaluate the kmeans cluster and plot the same
curve(dpand,"China") #call the curve function to plot the fit and data for China
curve(dpand,"Brazil") #call the curve function to plot the fit and data for China
agr(dpand, agri) #call agr function to compare agricultural land area and methane emission rate of Australia
