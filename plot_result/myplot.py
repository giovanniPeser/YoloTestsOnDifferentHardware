import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import os
from rich import print as rprint
from rich.panel import Panel
from matplotlib.ticker import ScalarFormatter
# from scipy.constants.constants import alpha

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

def getData(myarray, multi1000= True):
    toret = {}
    toret[Labels2[0]] =[]
    toret[Labels2[1]] =[]
    toret[Labels2[2]] =[]
    for i in range(0,len(myarray)):
        if(i%4==0):
            if (type(myarray[i])==type("1.0")):
                myarray[i] = myarray[i].replace(",",".")
            if(multi1000):
                myarray[i] = float(myarray[i])*int(1000)
            else:
                myarray[i] = float(myarray[i])
            toret[Labels2[i//8]].append(myarray[i])
    return toret

def plotData(data, name):
    fig4 = plt.figure()
    ax4 = plt.subplot()
    ax4.grid()
    ax4.set_ylabel("COCO mAP50-95")
    ax4.set_xlabel("Latency [ms/img]")
    if(name=="rasp"):
        myplot(ax4,data["cpu"], data["map50-95"], "-o", "CPU")
    else:
        myplot(ax4,data["cpu"], data["map50-95"], "-o", "GPU")
        myplot(ax4,data["gpu"], data["map50-95"], "-o", "GPU")
    ax4.legend(shadow=False, fontsize='medium',frameon=True,loc = 'lower right', prop={'size': 8})
    fig4.savefig(name+'_640.pdf',format='pdf',dpi = 1200)
    plt.show()

Labels = ["Y5s:640", "Y5s:320", "Qint8Y5s:640", "Qint8Y5s:320", "Y5n:640", "Y5n:320", "Qint8Y5n:640", "Qint8Y5n:320",
          "Y5su:640", "Y5su:320", "Qint8Y5su:640", "Qint8Y5su:320", "Y5nu:640", "Y5nu:320", "Qint8Y5nu:640", "Qint8Y5nu:320",
          "Y8s:640", "Y8s:320", "Qint8Y8s:640", "Qint8Y8s:320", "Y8n:640", "Y8n:320", "Qint8Y8n:640", "Qint8Y8n:320"]

Labels2 = ["YOLO5",
          "YOLO5U",
          "YOLO8"]

Labels3 = ["s", "n"]

filename = "result_for_plot.csv"
ex = pd.read_csv(filename, header=None, index_col=None)
filename = "result_precision.csv"
precision = pd.read_csv(filename, header=None, index_col=None)
         
nvidia = {"cpu": getData(np.array(ex[0][2:])), "gpu": getData(np.array(ex[1][2:])), "map50-95":getData(np.array(precision[1][1:]), False)}
dell = {"cpu": getData(np.array(ex[2][2:])), "gpu": getData(np.array(ex[3][2:])), "map50-95":getData(np.array(precision[1][1:]), False)}
raspberry = {"cpu": getData(np.array(ex[4][2:])), "map50-95":getData(np.array(precision[1][1:]), False)}

plotData(raspberry,"rasp")
plotData(dell,"dell")
plotData(nvidia,"nvidia")