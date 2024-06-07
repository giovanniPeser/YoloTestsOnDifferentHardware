import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np 
import os
from rich import print as rprint
from rich.panel import Panel
from matplotlib.ticker import ScalarFormatter

Labels = ["Y5s:640", "Y5s:320", "Qint8Y5s:640", "Qint8Y5s:320", "Y5n:640", "Y5n:320", "Qint8Y5n:640", "Qint8Y5n:320",
          "Y5su:640", "Y5su:320", "Qint8Y5su:640", "Qint8Y5su:320", "Y5nu:640", "Y5nu:320", "Qint8Y5nu:640", "Qint8Y5nu:320",
          "Y8s:640", "Y8s:320", "Qint8Y8s:640", "Qint8Y8s:320", "Y8n:640", "Y8n:320", "Qint8Y8n:640", "Qint8Y8n:320"]

Labels2 = ["YOLO5",
          "YOLO5U",
          "YOLO8"]

Labels3 = ["s", "n"]

def filterAll(myarray):
    for i in range(0,len(myarray)):
        if(myarray[i]=="/"):
            myarray[i] = None
        else:
            if (type(myarray[i])==type("1.0")):
                myarray[i] = myarray[i].replace(",",".")
            myarray[i] = float(myarray[i])*int(1000)
            print(myarray[i])
            #myarray[i] = float()*int(1000)


def myplot(ax, det_perf, map_perf, color, procUnit):
    i= 0
    for e in Labels2:
        print(det_perf[e])
        print(map_perf[e])
        ax.plot(det_perf[e], map_perf[e], color, label = Labels2[i] + " on "+ procUnit)
        for j in range(len(det_perf[e])):
            ax.annotate(Labels3[j], (det_perf[e][j], map_perf[e][j]))
        i=i+1

def get_value(myarray, i, multi1000):
    if (type(myarray)==type("1.0")):
        myarray = myarray.replace(",",".")
    if(multi1000):
        myarray = float(myarray)*int(1000)
    else:
        myarray = float(myarray)
    return myarray

def getData(myarray,isCpu=True, multi1000= True):
    toret = {}
    toret[Labels2[0]] =[]
    toret[Labels2[1]] =[]
    toret[Labels2[2]] =[]
    mylen = 24
    k = 0
    if(not(isCpu)):
        k = 4
    for i in range(0,mylen):
        if(i%4==0):
            temp = []
            temp.append(get_value(myarray[i][0+k],i,multi1000))
            temp.append(get_value(myarray[i][1+k],i,multi1000))
            temp.append(get_value(myarray[i][2+k],i,multi1000))
            toret[Labels2[i//8]].append(temp)
    return toret

def getPercentage(data,i):
    tot = data[i][0]+data[i][1]+data[i][2]
    return [data[i][0]/tot*100,data[i][1]/tot*100,data[i][2]/tot*100]
    

def getBarGroup(mystruc, procUnit, index):
    graph1 = getPercentage(mystruc[procUnit]["YOLO5"],index)
    graph2 = getPercentage(mystruc[procUnit]["YOLO5U"],index)
    graph3 = getPercentage(mystruc[procUnit]["YOLO8"],index)
    graphs = [graph1,graph2, graph3]
    prepro = []
    infe = []
    postpro = []
    for i in range(0,len(graphs)):
        prepro.append(graphs[i][0])
        infe.append(graphs[i][1])
        postpro.append(graphs[i][2])
    allphase = [prepro,infe,postpro]
    return allphase

"""
Commands to Plot Raspberry
"""
def plotRasp(rasp):

    sPhase =  getBarGroup(rasp,"cpu",0)
    nPhase =  getBarGroup(rasp,"cpu",1)


    fig = plt.figure(constrained_layout=True)

    # Values for plotting
    yoloLabels = ["YOLO5", "YOLO5U", "YOLO8"]
    subLabels = ["s", "n"]
    fullLabels = [f"{y}({s})" for y in yoloLabels for s in subLabels]  # Combine yoloLabels and subLabels
    x2 = np.arange(len(fullLabels))  

    labels= ["Pre-processing", "Inference", "Post-processing"]
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, ax = plt.subplots()
    bottom1 = np.zeros(3)
    bottom2 = np.zeros(3)
    # Plot the graph
    for i in range(0,len(labels)):
        p1 = ax.bar(x-0.2, sPhase[i], width, label=labels[i], bottom=bottom1, color=colors[i])
        bottom1 += sPhase[i]
        p2 = ax.bar(x+0.2, nPhase[i], width, bottom=bottom2, color=colors[i])
        bottom2 += nPhase[i]

    # Adding labels, title, and legend
    ax.set_ylabel('Percentage')
    ax.set_title('Mean Elapsed Time Distribution on Dell XPS 15 (GPU)')
    x1 = list(x-0.2)
    x2 = list(x+0.2)
    totx = x1+x2
    totx.sort()
    ax.set_xticks(totx)
    ax.set_xticklabels(fullLabels, fontsize=9)
    ax.legend(loc = 'center', prop={'size': 11},ncol=3)

    # Adjust layout
    #fig.tight_layout()

    # Save or show the plot
    plt.savefig('raspYoloN640DistrBars.pdf', format='pdf', dpi=2400)
    plt.show()



"""
Commands to Plot or Dell or Nvidia
"""

def plotDellOrNvidia(data, name):
    sPhaseCPU =  getBarGroup(data,"cpu",0)
    nPhaseCPU =  getBarGroup(data,"cpu",1)
    sPhaseGPU =  getBarGroup(data,"gpu",0)
    nPhaseGPU =  getBarGroup(data,"gpu",1)
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(nrows=2, ncols=1)
    axs0 = subfigs[0].subplots(nrows=1, ncols=1)
    axs1 = subfigs[1].subplots(nrows=1, ncols=1)
    # Values for plotting
    yoloLabels = ["YOLO5", "YOLO5U", "YOLO8"]
    subLabels = ["s", "n"]
    fullLabels = [f"{y}({s})" for y in yoloLabels for s in subLabels]  # Combine yoloLabels and subLabels
    x2 = np.arange(len(fullLabels))  

    labels= ["Pre-processing", "Inference", "Post-processing"]
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    #mpl.style.use("seaborn-pastel")
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    bottom1 = np.zeros(3)
    bottom2 = np.zeros(3)
    #Plot First subgraph
    for i in range(0,len(labels)):
        p1 = axs0.bar(x-0.2, sPhaseCPU[i], width, label=labels[i], bottom=bottom1, color=colors[i])
        bottom1 += sPhaseCPU[i]
        p2 = axs0.bar(x+0.2, nPhaseCPU[i], width, bottom=bottom2, color=colors[i])
        bottom2 += nPhaseCPU[i]

    bottom1 = np.zeros(3)
    bottom2 = np.zeros(3)
    # Plot Second subgraph
    for i in range(0,len(labels)):
        p1 = axs1.bar(x-0.2, sPhaseGPU[i], width, label=labels[i], bottom=bottom1, color=colors[i])
        bottom1 += sPhaseGPU[i]
        p2 = axs1.bar(x+0.2, nPhaseGPU[i], width, bottom=bottom2, color=colors[i])
        bottom2 += nPhaseGPU[i]

    # Adding labels, title, and legend
    myfontsize=10
    axs0.set_ylabel('Percentage')
    axs0.set_title('Using CPU', fontsize=myfontsize)
    x1 = list(x-0.2)
    x2 = list(x+0.2)
    totx = x1+x2
    totx.sort()
    axs0.set_xticks(totx)
    axs0.set_xticklabels(fullLabels, fontsize=myfontsize)

    axs1.set_ylabel('Percentage')
    axs1.set_title('Using GPU', fontsize=myfontsize)
    axs1.set_xticks(totx)
    axs1.set_xticklabels(fullLabels, fontsize=myfontsize)
    handles, labels = axs0.get_legend_handles_labels()
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    axs0.legend(handles, labels, loc = 'center', prop={'size': 11},ncol=3)

    # Save or show the plot
    plt.savefig(name +'AllYoloN640DistrBars.pdf', format='pdf', dpi=2400)
    plt.show()


filename = "nvidia.csv"
nvidia_data = pd.read_csv(filename, header=None, index_col=None)
nvidia = {"cpu": getData(np.array(nvidia_data[0:][2:])), "gpu": getData(np.array(nvidia_data[0:][2:]),False)}
filename = "dell.csv"
dell_data = pd.read_csv(filename, header=None, index_col=None)
dell = {"cpu": getData(np.array(dell_data[0:][2:])), "gpu": getData(np.array(dell_data[0:][2:]),False)}
rasp_data = pd.read_csv(filename, header=None, index_col=None)
rasp = {"cpu": getData(np.array(nvidia_data[0:][2:]))}
myexplode = [0, 0.2, 0.2]

plotDellOrNvidia(nvidia, "nvidia")
plotDellOrNvidia(dell, "dell")
plotRasp(rasp)
