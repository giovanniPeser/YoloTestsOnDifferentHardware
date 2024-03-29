import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import os
from rich import print as rprint
from rich.panel import Panel
from matplotlib.ticker import ScalarFormatter
# from scipy.constants.constants import alpha

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
        

filename = "result_for_plot.csv"
ex = pd.read_csv(filename, header=None, index_col=None)

filename = "result_precision.csv"
precision = pd.read_csv(filename, header=None, index_col=None)

            
nvidia = {"cpu": getData(np.array(ex[0][2:])), "gpu": getData(np.array(ex[1][2:])), "map50-95":getData(np.array(precision[1][1:]), False)}
dell = {"cpu": getData(np.array(ex[2][2:])), "gpu": getData(np.array(ex[3][2:])), "map50-95":getData(np.array(precision[1][1:]), False)}
raspberry = {"cpu": getData(np.array(ex[4][2:])), "map50-95":getData(np.array(precision[1][1:]), False)}

"""
filterAll(nvidia["cpu"])
filterAll(nvidia["gpu"])
filterAll(dell["cpu"])
filterAll(dell["cpu"])
filterAll(raspberry["cpu"])
"""
print(nvidia)
print(dell)
print(raspberry)

fig4 = plt.figure()
#yfmt = ScalarFormatter()
#yfmt.set_powerlimits((0,1)) 
ax4 = plt.subplot()
ax4.grid()
#ax4.set_xlim([4,60])
#ax4.set_title("Performances on Nvidia Jetson Nano with size=640")
ax4.set_ylabel("COCO mAP50-95")
ax4.set_xlabel("Latency [ms/img]")
#ax4.yaxis.set_major_formatter(yfmt)
#ax4.plot(nvidia["cpu"][Labels2[0]] ,nvidia["map50-95"][Labels2[0]],'-o', label = Labels2[0] + " on CPU")
#ax4.plot(nvidia["cpu"][Labels2[1]] ,nvidia["map50-95"][Labels2[1]],'-o', label = Labels2[1] + " on CPU")
#ax4.plot(nvidia["cpu"][Labels2[1]] ,nvidia["map50-95"][Labels2[2]],'-o', label = Labels2[2] + " on CPU")
myplot(ax4,raspberry["cpu"], raspberry["map50-95"], "-o", "CPU")
#myplot(ax4,dell["gpu"], dell["map50-95"], "-o", "GPU")

#ax4.plot(nvidia["gpu"][Labels2[0]] ,nvidia["map50-95"][Labels2[0]],'-o', label = Labels2[0] + " on GPU")
#ax4.plot(nvidia["gpu"][Labels2[1]] ,nvidia["map50-95"][Labels2[1]],'-o', label = Labels2[1] + " on GPU")
#ax4.plot(nvidia["gpu"][Labels2[1]] ,nvidia["map50-95"][Labels2[2]],'-o', label = Labels2[2] + " on GPU")
#ax4.plot(snr6,per6,label = '54 Mbps with Nist 500 byte NEW',color='r',LineWidth = 1,LineStyle = '-') 
#ax4.plot(snr5,per5,label = '54 Mbps with Nist 500 byte',color='b',LineWidth = 1,LineStyle = '-') 
#ax4.plot(snr6_n ,per6_n,label = '6 Mbps with Nist 50 byte NEW',color='y',LineWidth = 1,LineStyle = '-') 
#ax4.plot(nvidia["cpu"] ,nvidia["map50-95"],'-ro', label = 'Nvidia CPU')#,LineStyle = '-',marker='*',markevery=5)
#ax4.plot(power ,np.array(udp_mean)-np.array(udp_std),LineWidth = 1,color='C1',LineStyle = '-.',alpha = 0.6)#,marker='*',markevery=5)
#ax4.plot(snr6_o,per6_o,label = '6 Mbps with Nist 50byte',color='y',LineWidth = 1,LineStyle = '-') 
legend = ax4.legend(shadow=False, fontsize='medium',frameon=True,loc = 'lower right', prop={'size': 8})
fig4.savefig('rasp_640.pdf',format='pdf',dpi = 1200)
plt.show()