import os
import numpy as np
import matplotlib.pyplot as plt


with open('click5000/clickresultnaive.txt', 'r') as f1, open('click5000/clickresultclip.txt', 'r') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    naive = []
    bal = []
    num = len(lines1)
    prop1 = 0
    prop2 = 0
    diff_num = 0
    for i in range(num):
        prop1 += float(lines1[i])
        prop2 += float(lines2[i])
        naive.append(float(lines1[i]))
        bal.append(float(lines2[i]))
        if abs(float(lines1[i]) - float(lines2[i])) > 1e-10:
            diff_num += 1
print(sum([naive[i] <= bal[i] and naive[i]<=0.1 for i in range(5000)]))
print(sum([naive[i]<=0.1 for i in range(5000)]))
print(sum([bal[i] <= naive[i] and bal[i]<=0.1 for i in range(5000)]))
print(sum([bal[i]<=0.1 for i in range(5000)]))
print("naive: {}".format(prop1/num))
print("bal: {}".format(prop2/num))
print(diff_num)
print(num)
# plt.hist(naive, bins=20)
# plt.show()
# plt.hist(bal, bins=20)
# plt.show()
