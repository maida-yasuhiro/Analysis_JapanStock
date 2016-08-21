import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.cluster import KMeans

DATE = 30
csvfiles = pd.read_csv('meigara_csv.csv')
csvfiles = pd.DataFrame(csvfiles)
csvfiles = list(csvfiles.values.flatten())

names = pd.read_csv('meigara.csv')
names = pd.DataFrame(names)
names = list(names.values.flatten())

def getAdjClose(csvfile):
    stock = pd.read_csv("./kabuka/" + csvfile,index_col=0, parse_dates=True)
    stock_close = stock['Adj Close'].tail(DATE)
    returns = pd.Series(stock_close).pct_change()
    ret_index = (1 + returns).cumprod() # 累積積を求める
    ret_index[0] = 1 # 最初の値を 1.0 にする
    return ret_index

def getDate():
  stock = pd.read_csv("./kabuka/" + csvfiles[0],index_col=0, parse_dates=True)
  dates = stock.tail(DATE).index.values
  return dates

def kmeans(features):
    # k=3, ランダマイズを 10 回実施する
    kmeans_model = KMeans(n_clusters=6, random_state=15).fit(features)
    # ラベルを取り出す
    print(kmeans_model)
    print(kmeans_model.labels_)
    labels = kmeans_model.labels_
    return labels

def figureplot(fig,figname):
  plt.figure()
  fig.plot()
  plt.subplots_adjust(bottom=0.20)
  plt.legend(loc="best",prop={'size' : 8})
  plt.savefig(figname)
  plt.close()


list = []
for csvfile in csvfiles:
     list.append(getAdjClose(csvfile))

dates = getDate()

list = np.array(list)
#list = list.reshape((DATE,len(list)))

list = pd.DataFrame(list.T,index = dates, columns = names)
list[np.isnan(list)] = np.nanmean(list)
print (list)

a0 = 0
a1 = 0
a2 = 0
a3 = 0
a4 = 0
a5 = 0
a6 = 0
a7 = 0
a8 = 0
list0 = pd.DataFrame([],index = dates)
list1 = pd.DataFrame([],index = dates)
list2 = pd.DataFrame([],index = dates)
list3 = pd.DataFrame([],index = dates)
list4 = pd.DataFrame([],index = dates)
list5 = pd.DataFrame([],index = dates)
list6 = pd.DataFrame([],index = dates)
list7 = pd.DataFrame([],index = dates)
list8 = pd.DataFrame([],index = dates)
labels = kmeans(list.T)
for name,label,feature in zip(names,labels, list):
    print(label, name)
    if label == 0:
      list0[name] = list[name] 
      list0[np.isnan(list0)] = np.nanmean(list0)
      a0 += 1
    elif label == 1:
      list1[name] = list[name] 
      list1[np.isnan(list1)] = np.nanmean(list1)
      a1 += 1
    elif label == 2:
      list2[name] = list[name] 
      list2[np.isnan(list2)] = np.nanmean(list2)
      a2 += 1
    elif label == 3:
      list3[name] = list[name] 
      list3[np.isnan(list3)] = np.nanmean(list3)
      a3 += 1
    elif label == 4:
      list4[name] = list[name] 
      list4[np.isnan(list4)] = np.nanmean(list4)
      a4 += 1
    elif label == 5:
      list5[name] = list[name] 
      list5[np.isnan(list5)] = np.nanmean(list5)
      a5 += 1
    elif label == 6:
      list6[name] = list[name] 
      list6[np.isnan(list6)] = np.nanmean(list6)
      a6 += 1
    elif label == 7:
      list7[name] = list[name] 
      list7[np.isnan(list7)] = np.nanmean(list7)
      a7 += 1
    elif label == 8:
      list8[name] = list[name] 
      list8[np.isnan(list8)] = np.nanmean(list8)
      a8 += 1

if a0 != 0:
  figureplot(list0,"picture/cluster0_30days.png")

if a1 != 0:
  figureplot(list1,"picture/cluster1_30days.png")

if a2 != 0:
  figureplot(list2,"picture/cluster2_30days.png")

if a3 != 0:
  figureplot(list3,"picture/cluster3_30days.png")

if a4 != 0:
  figureplot(list4,"picture/cluster4_30days.png")

if a5 != 0:
  figureplot(list5,"picture/cluster5_30days.png")

if a6 != 0:
  figureplot(list6,"picture/cluster6_30days.png")

if a7 != 0:
  figureplot(list7,"picture/cluster7_30days.png")

if a8 != 0:
  figureplot(list8,"picture/cluster8_30days.png")
