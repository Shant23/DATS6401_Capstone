import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from scipy.stats import chi2

def ADF_cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def plotACF(x, k):
    Xp = list(range(0, k))
    Xr = [-x for x in Xp]
    Xr = Xr[::-1]
    Xr = np.array(Xr[:-1])
    Xp = np.concatenate([Xr, Xp])

    Yp = x
    Yr = Yp[::-1]
    Yr = np.array(Yr[:-1])
    Yp = np.concatenate([Yr, Yp])

    plt.figure()
    plt.stem(Xp, Yp, use_line_collection=True)
    plt.title("stem figure of autocorrelation")
    plt.xlabel("number of lags")
    plt.ylabel('correlation')
    plt.show()


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

data1 = pd.read_csv('C:/Users/shant/Documents/Hosp Data/term project/readmissionsonly.csv')
readmissions=data1.iloc[:,1]
rea=readmissions.values

ADF_cal(rea)

rea=data1.iloc[:,1]
submeanS= list()
subvarS = list()
for i in range(1, len(rea)):
    s = list()
    v = list()
    s = np.mean(rea.head(i))
    v = np.var(rea.head(i))
    submeanS.append(s)
    subvarS.append(v)

plt.figure(1)
plt.plot(submeanS)
plt.title("Mean of Readmissions over time")
plt.xlabel("Time")
plt.ylabel("Readmissions mean over time")
plt.show()


plt.figure(2)
plt.plot(subvarS)
plt.title("Variance of Readmissions over time")
plt.xlabel("Time")
plt.ylabel("Readmissions variance over time")
plt.show()


y1=acf(rea, fft=True, nlags=19)
y=y1.tolist()
plotACF(y, 20)
lags=20




rea1=difference(rea)
rea=rea1

submeanS = list()
subvarS = list()
for i in range(1, len(rea)):
    s = list()
    v = list()
    s = np.mean(rea.head(i))
    v = np.var(rea.head(i))
    submeanS.append(s)
    subvarS.append(v)

plt.figure(1)
plt.plot(submeanS)
plt.title("Mean of First Differenced data over time")
plt.xlabel("Time")
plt.ylabel("First differenced mean over time")
plt.show()


plt.figure(2)
plt.plot(subvarS)
plt.title("Variance of first differenced data over time")
plt.xlabel("Time")
plt.ylabel("First differenced variance over time")
plt.show()

ADF_cal(rea)
y1=acf(rea, fft=True, nlags=19)
y=y1.tolist()
plotACF(y, 20)




phi = np.zeros(shape=[9, 10])
for k in range(1, 10):
    for j in range(0, 9):
        if k == 1:
            phih = y[j + 1] / y[j]
        else:
            A = []
            a = y[j:j + k]
            b = a[:]
            for i in range(len(a)):
                A.append(b)
                b = a[i + 1:i + 2] + b[:-1]
            A = np.array(A)
            y2 = A[:, :-1]
            y3 = np.array([y[j + 1:k + j + 1]]).T
            y4 = np.hstack((y2, y3))
            num = np.linalg.det(y4)
            den = np.linalg.det(A)
            phih = num / den
        phi[j][k] = phih

table=pd.DataFrame(phi)
table=table.drop(table.columns[0], axis=1)
table=np.round(table, decimals=3)
print(table)

#Iteration of all different orders for ARMA with creation of a Q table and an AIC table

Q1=np.zeros(shape=[9])
Q2=np.zeros(shape=[9, 9])
AIC1=np.zeros(shape=[9])
AIC2=np.zeros(shape=[9, 9])
for na in range(0, 9):
    if na == 0:
        for nb in range(1,9):
            try:
                model = sm.tsa.ARMA(rea, (na, nb)).fit(trend='nc', disp=0)
                model_hat = model.predict(start=0, end=(len(rea) - 1))
                e = rea - model_hat
                resacf = acf(e, fft=True, nlags=19)
                Qr = len(rea) * np.sum(np.square(resacf[1:]))
                Q1[nb]=Qr
                AICl[nb]=model.aic
            except:
                pass
    else:
        for nb in range(0,9):
            try:
                model = sm.tsa.ARMA(rea, (na, nb)).fit(trend='nc', disp=0)
                model_hat = model.predict(start=0, end=(len(rea) - 1))
                e = rea - model_hat
                resacf = acf(e, fft=True, nlags=19)
                Qr = len(rea) * np.sum(np.square(resacf[1:]))
                Q2[nb][na]=Qr
                AIC2[nb][na]=model.aic
            except:
                pass
table1=pd.DataFrame(Q2)
table1=np.round(table1, decimals=4)
table2=pd.DataFrame(AIC2)
table3=np.round(table2, decimals=4)
print(table1)
print(table2)
print(Q1)
print(AIC1)
table3.to_csv('AICtable1.csv')



na=4
nb=8


model = sm.tsa.ARMA(rea,(na,nb)).fit(trend='nc',disp=0)
for i in range(na):
    print("The AR coefficient a{}".format(i), "is:", model.params[i])
for i in range(nb):
    print("The MA coefficient b{}".format(i), "is:", model.params[i+na])
print(model.summary())

model_hat = model.predict(start=0, end=(len(rea)-1))

plt.figure()
plt.plot(rea,'r', label = "True data")
plt.plot(model_hat,'b', label = "Fitted data")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.legend()
plt.title("fitted vs true values")
plt.show()



e =rea-model_hat

resacf= acf(e, fft=True, nlags=19)

plotACF(resacf , 20)

print('the mean of the residuals is: {}'.format(np.mean(e)))
print('the variance of the residuals is: {}'.format((np.std(e)*np.std(e))))


Q = len(rea)*np.sum(np.square(resacf[1:]))
DOF = lags - na - nb
chi_critical = chi2.ppf(0.95, DOF)
if Q< chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")
lbvalue, pvalue, bpvalue, bppvalue = sm.stats.acorr_ljungbox(e, lags=[19], boxpierce=True)
print(lbvalue)
print(pvalue)
print(bpvalue)
print(bppvalue)
print(np.sqrt(np.mean(e**2)))
print(Q)
meany=np.mean(rea)
R2sq2 = 1 - (np.sum((e) ** 2)) / (np.sum((rea - meany) ** 2))
print("the R squared for this model is:{}".format(R2sq2))

na=8
nb=6


model = sm.tsa.ARMA(rea,(na,nb)).fit(trend='nc',disp=0)
for i in range(na):
    print("The AR coefficient a{}".format(i), "is:", model.params[i])
for i in range(nb):
    print("The MA coefficient b{}".format(i), "is:", model.params[i+na])
print(model.summary())

model_hat = model.predict(start=0, end=(len(rea)-1))

plt.figure()
plt.plot(rea,'r', label = "True data")
plt.plot(model_hat,'b', label = "Fitted data")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.legend()
plt.title("fitted vs true values")
plt.show()



e =rea-model_hat

resacf= acf(e, fft=True, nlags=19)

plotACF(resacf , 20)

print('the mean of the residuals is: {}'.format(np.mean(e)))
print('the variance of the residuals is: {}'.format((np.std(e)*np.std(e))))


Q = len(rea)*np.sum(np.square(resacf[1:]))
DOF = lags - na - nb
chi_critical = chi2.ppf(0.95, DOF)
if Q< chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")
lbvalue, pvalue, bpvalue, bppvalue = sm.stats.acorr_ljungbox(e, lags=[19], boxpierce=True)
print(lbvalue)
print(pvalue)
print(bpvalue)
print(bppvalue)
meany=np.mean(rea)
R2sq2 = 1 - (np.sum((e) ** 2)) / (np.sum((rea - meany) ** 2))
print("the R squared for this model is:{}".format(R2sq2))
print(Q)
