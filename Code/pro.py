#PROGRAM FOR VISUALISING (DIS)AGRREMENT BETWEEN TWO ANNOTATIONS
import matplotlib.pyplot as plt
import csv

input_file = csv.DictReader(open("1.csv"))          #10mins to data from file Portband_21
A = []          #Array to store A's annotation
B = []          #Array to store B's annotation
X = []          #Array for X axis, basically time.
Y_A =[]         #Array for storing Agreements


for row in input_file:
    mylist = row['StudioEventTypeDiff'].split('_')
        # StudioEventTypeDiff takes values like Disagree_4_MediaView_5_Unknown OR Agree_1_Scanning
        # Following are the values associated with each categoy: 1 Scanning;  2 Skimming;  3 Reading; 4 MediaView; 5 Unknow
        # These values are reflected in graph on Y axis.

    if mylist[0] == 'Agree' and ( len(mylist) >= 3  ):
        A.append(int(mylist[1]) )
        B.append(int(mylist[1]) + 0.05)         #Values of B's annotation were shifted up to have some sort of clarity in graph.
        Y_A.append(0.5)                         #Storing Agreements

    elif mylist[0] == 'Disagree':
        A.append(int(mylist[1])    )
        B.append(int(mylist[3]) + 0.05 )
        Y_A.append(0)



for i in range(len(A)):
    X.append(i)


for i in [1,2,3]:
    if i==1:
        plt.plot(X,A,'r')
    elif i==2:
        plt.plot(X,B,'b')
    else:
        plt.plot(X,Y_A,'g')

plt.show()
