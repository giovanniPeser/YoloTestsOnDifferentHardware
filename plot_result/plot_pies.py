import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import os
from rich import print as rprint
from rich.panel import Panel
from matplotlib.ticker import ScalarFormatter

Labels = ["Y5s:640", "Y5s:320", "Qint8Y5s:640", "Qint8Y5s:320", "Y5n:640", "Y5n:320", "Qint8Y5n:640", "Qint8Y5n:320",
          "Y5su:640", "Y5su:320", "Qint8Y5su:640", "Qint8Y5su:320", "Y5nu:640", "Y5nu:320", "Qint8Y5nu:640", "Qint8Y5nu:320",
          "Y8s:640", "Y8s:320", "Qint8Y8s:640", "Qint8Y8s:320", "Y8n:640", "Y8n:320", "Qint8Y8n:640", "Qint8Y8n:320"]

Labels2 = ["Yolo5",
          "Yolo5U",
          "Yolo8"]

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
        k = 3
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
    

filename = "nvidia.csv"
nvidia_data = pd.read_csv(filename, header=None, index_col=None)
nvidia = {"cpu": getData(np.array(nvidia_data[0:][2:])), "gpu": getData(np.array(nvidia_data[0:][2:]),False)}
filename = "dell.csv"
dell_data = pd.read_csv(filename, header=None, index_col=None)
dell = {"cpu": getData(np.array(dell_data[0:][2:])), "gpu": getData(np.array(dell_data[0:][2:]),False)}
rasp_data = pd.read_csv(filename, header=None, index_col=None)
rasp = {"cpu": getData(np.array(nvidia_data[0:][2:]))}

myexplode = [0, 0.2, 0.2]
"""
graph1 = getPercentage(rasp["cpu"]["Yolo5"],1)
graph2 = getPercentage(rasp["cpu"]["Yolo5U"],1)
graph3 = getPercentage(rasp["cpu"]["Yolo8"],1)
fig = plt.figure(constrained_layout=True)

labels= ["Pre-processing", "Inference", "Post-processing"]
axs0 = fig.subplots(nrows=1, ncols=3)
axs0[0].pie(graph1, explode = myexplode)
axs0[0].set_title('Yolo 5')
axs0[1].pie(graph2, explode = myexplode)
axs0[1].set_title('Yolo 5U')
axs0[2].pie(graph3, explode = myexplode)
axs0[2].set_title('Yolo 8')
# show plot
fig.legend(labels=labels, loc = 'upper center', prop={'size': 8},ncol=3)
fig.savefig('raspYoloN640Distr.pdf',format='pdf',dpi = 2400)
plt.show()

"""
graph1 = getPercentage(dell["cpu"]["Yolo5"],1)
graph2 = getPercentage(dell["cpu"]["Yolo5U"],1)
graph3 = getPercentage(dell["cpu"]["Yolo8"],1)
graph4 = getPercentage(dell["gpu"]["Yolo5"],1)
graph5 = getPercentage(dell["gpu"]["Yolo5U"],1)
graph6 = getPercentage(dell["gpu"]["Yolo8"],1)
#fig = plt.figure()
fig = plt.figure(constrained_layout=True)

# create 3x1 subfigs
labels= ["Pre-processing", "Inference", "Post-processing"]
subfigs = fig.subfigures(nrows=2, ncols=1)
axs0 = subfigs[0].subplots(nrows=1, ncols=3)
axs1 = subfigs[1].subplots(nrows=1, ncols=3)
subfigs[0].suptitle('Using CPU')
subfigs[1].suptitle('Using GPU')
axs0[0].pie(graph1, explode = myexplode)
axs0[0].set_title('Yolo 5')
axs0[1].pie(graph2, explode = myexplode)
axs0[1].set_title('Yolo 5U')
axs0[2].pie(graph3, explode = myexplode)
axs0[2].set_title('Yolo 8')
axs1[0].pie(graph4, explode = myexplode)
axs1[0].set_title('Yolo 5')
axs1[1].pie(graph5, explode = myexplode)
axs1[1].set_title('Yolo 5U')
axs1[2].pie(graph6, explode = myexplode)
axs1[2].set_title('Yolo 8')
# show plot
fig.legend(labels=labels, loc = 'lower center', prop={'size': 8},ncol=3)
fig.savefig('dellYoloN640Distr.pdf',format='pdf',dpi = 2400)
plt.show()

"""
fig, axs= plt.subplots(2,3)
# create 3x1 subfigs
subfigs = fig.subfigures(nrows=2, ncols=1)
subfigs[0].suptitle('Using CPU')
subfigs[1].suptitle('Using GPU')
labels= ["Pre-processing", "Inference", "NMS"]
axs[0][0].pie(graph1, explode = myexplode)
axs[0][0].set_title('Yolo 5')
axs[0][1].pie(graph2)
axs[0][1].set_title('Yolo 5U')
axs[0][2].pie(graph3)
axs[0][2].set_title('Yolo 8')
axs[1][0].pie(graph4)
axs[1][0].set_title('Yolo 5')
axs[1][1].pie(graph5)
axs[1][1].set_title('Yolo 5U',fontsize=2)
axs[1][2].pie(graph6)
axs[1][2].set_title('Yolo 8')
# show plot
fig.legend(labels=labels, loc = 'lower center', prop={'size': 8})
plt.show()
"""
"""
print("Nvidia")
print(getPercentage(nvidia["cpu"]["Yolo5"],1))
print(getPercentage(nvidia["cpu"]["Yolo5U"],1))
print(getPercentage(nvidia["cpu"]["Yolo8"],1))
print(getPercentage(nvidia["gpu"]["Yolo5"],1))
print(getPercentage(nvidia["gpu"]["Yolo5U"],1))
print(getPercentage(nvidia["gpu"]["Yolo8"],1))
print("--------------------------------")
print("Dell")
print(getPercentage(dell["cpu"]["Yolo5"],1))
print(getPercentage(dell["cpu"]["Yolo5U"],1))
print(getPercentage(dell["cpu"]["Yolo8"],1))
print(getPercentage(dell["gpu"]["Yolo5"],1))
print(getPercentage(dell["gpu"]["Yolo5U"],1))
print(getPercentage(dell["gpu"]["Yolo8"],1))
print("--------------------------------")
print("Rasp")
print(getPercentage(rasp["cpu"]["Yolo5"],1))
print(getPercentage(rasp["cpu"]["Yolo5U"],1))
print(getPercentage(rasp["cpu"]["Yolo8"],1))
"""