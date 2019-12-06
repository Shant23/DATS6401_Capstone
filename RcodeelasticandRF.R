library(ROSE)
library(glmnet)
library(ISLR)
library(plotmo)
library(gradDescent)
library(caret)
library(dplyr)
library(randomForest)
library(e1071)
require(useful)
library(ggplot2)
library(readxl)

#Loading the dataset and defining the target

setwd('C:/Users/shant/Documents/Hosp Data/term project')
data1=read.csv("matrixformatfinal.csv")
data2 = data1[,-3] 
data3=model.matrix(~., data=data2)
data4=data.frame(data3)
target=data1$test

#create the data split into test and train with 80/20

prop_split = 0.8 
train_index = sample(1:n_obs, round(n_obs * prop_split))

#Create train and test
trainset = data3[train_index,]
testset = data3[-train_index,]
trainset=as.matrix(trainset)
testset=as.matrix(testset)
traintarget = target[train_index]
testtarget = target[-train_index]
testtarget=as.factor(testtarget)
testtarget2=as.numeric(testtarget)

#Create first model (model1) with entire dataset. This will yeild all of the non zero coefficients which are saved 
#on a csv file.
cv.fit <- cv.glmnet(x = trainset, y = traintarget, family = "binomial", type.measure = "auc", nfold=5)
cvfitcoeff=coef(cv.fit, s="lambda.1se")
cvfitcoeff
coeff= as.matrix(coef(cv.fit, s="lambda.1se"))
coefDF=data.frame(Value=coeff, coefficient=rownames(coeff))
coefDF=coefDF[nonzeroCoef(coef(cv.fit, s="lambda.1se")), ]
write.csv(coefDF, "coefDF.csv")

#Diagnositics were run to evalaute the data. 
y_hat_elastic <- predict(cv.fit, testset, s="lambda.1se", type="class")
RMSE1= sqrt(mean((as.numeric(y_hat_elastic))-testtarget2)^2)
table(y_hat_elastic, testtarget )
table(data1$test)
plot(cv.fit)

#The reduced dataset was created and then will be used for future evalaution.
#Another cross validated elastic net was run to identify the most important features. 
df1=read.csv('reducedtestaddedR.csv')
n_obs2 = dim(df1)

#Remove the target and define
df2=df1[,-183]
target3=df1$test

#create the model.matrix and the train and test data, model2
prop_split = 0.8 
train_index = sample(1:n_obs2, round(n_obs2 * prop_split))
df3=model.matrix(~., data=df2)
trainset3 = df3[train_index,]
testset3 = df3[-train_index,]
traintarget3 = target[train_index]
testtarget3 = target[-train_index]


#test model2
cv.fit2 <- cv.glmnet(x = trainset3, y = traintarget3, family = "binomial", type.measure = "auc", nfold=5)
yhat2=predict(cv.fit2, testset3, s="lambda.1se", type="class")
table(yhat2, testtarget3)
RMSE2= sqrt(mean((as.numeric(yhat2))-testtarget3)^2)
yhat2=as.numeric(yhat2)
testtarget3=as.numeric(testtarget3)
plot(cv.fit2)



cvfitcoeff2=coef(cv.fit2, s="lambda.1se")
cvfitcoeff2
coeff2= as.matrix(coef(cv.fit, s="lambda.1se"))
coefDF2=data.frame(Value=coeff2, coefficient=rownames(coeff2))
coefDF2=coefDF2[nonzeroCoef(coef(cv.fit, s="lambda.1se")), ]
write.csv(coefDF2, "coefDF2.csv")

#create smaller dataset for model3
secondtrainX=df2%>%select(Admitting_Source_OUTPATIENT_CLINIC, Admitting_Source_ROUTINE_IP_ADMISSION__UNSCHEDULED_,
                          Admitting_Source_TRANS_SKILLED_NURSING_FACILITY, Principal_Payer_AMERIHEALTH_ALLNCE,
                          Discharge_APR_DRG_MDC23,
                          DRG_MDC_NO_17, Admitting_Loc_HOSP_ER_HOLDING, Discharge_APR_DRG_MDC18,
                          Discharge_APR_DRG_MDC16,
                          Facility_DC_Disposition_DISCH_TO_HOSPICE_HOME__CONTINUOUS_CARE_,
                          Principal_Payer_H_S_C_S_N__MCAID_HMO,Principal_Payer_MEDICARE_PT_B_IP, lymph,
                          metacanc,
                          Admitting_Loc_GWU_5_SO_ONCOLOGY, solidtum,
                          Principal_Payer_AMERIHEALTH_MCAID,
                          Principal_Payer_MEDICAID, diabc, rheumd,
                          Discharge_APR_DRG_MDC11, pvd, CDB_APR_DRG_Acute_Care_Expected_Mortality, Admitting_Loc_GWU_PACU,
                          Principal_Payer_NON_CONTRCTD_HMO_PPO, CDB_APR_DRG_All_Inpatient_Expected_Mortality,
                          Admitting_Service_Code_NEU,
                          Principal_Payer_Type_SELF_PAY_ADMIT)
                          

#Create test and train for model3                          
n_obs3 = dim(secondtrainX)
colnames(secondtrainX)
prop_split = 0.8 
train_index = sample(1:n_obs3, round(n_obs3 * prop_split)) 

target4=df1$test
secondtrainX2=model.matrix(~., data=secondtrainX)
trainset4 = secondtrainX2[train_index,]
testset4 = secondtrainX2[-train_index,]
traintarget4 = target4[train_index]
testtarget4 = target4[-train_index]

#Train and test model3
cv.fit3 <- cv.glmnet(x = trainset4, y = traintarget4, family = "binomial", type.measure = "auc", nfold=5)
yhat3=predict(cv.fit3, testset4, s="lambda.1se", type="class")
table(yhat3, testtarget3)
RMSE3= sqrt(mean((as.numeric(y_hat_elastic))-testtarget4)^2)
RMSE3

plot(cv.fit3)

cvfitcoeff2=coef(cv.fit2, s="lambda.1se")
cvfitcoeff2
coeff2= as.matrix(coef(cv.fit, s="lambda.1se"))
coefDF2=data.frame(Value=coeff2, coefficient=rownames(coeff2))
coefDF2=coefDF2[nonzeroCoef(coef(cv.fit, s="lambda.1se")), ]
write.csv(coefDF2, "coefDF2.csv")

                          
#generate a random forest model for predictions
#set the seed
set.seed(71) 
#create the randomforest model
#start with the dataset:
setrf1= df1%>%select(Admitting_Source_OUTPATIENT_CLINIC, Admitting_Source_ROUTINE_IP_ADMISSION__UNSCHEDULED_,
             Admitting_Source_TRANS_SKILLED_NURSING_FACILITY, Principal_Payer_AMERIHEALTH_ALLNCE,
             Discharge_APR_DRG_MDC23,
             DRG_MDC_NO_17, Admitting_Loc_HOSP_ER_HOLDING, Discharge_APR_DRG_MDC18,
             Discharge_APR_DRG_MDC16,
             Facility_DC_Disposition_DISCH_TO_HOSPICE_HOME__CONTINUOUS_CARE_,
             Principal_Payer_H_S_C_S_N__MCAID_HMO,Principal_Payer_MEDICARE_PT_B_IP, lymph,
             metacanc,
             Admitting_Loc_GWU_5_SO_ONCOLOGY, solidtum,
             Principal_Payer_AMERIHEALTH_MCAID,
             Principal_Payer_MEDICAID, diabc, rheumd,
             Discharge_APR_DRG_MDC11, pvd, CDB_APR_DRG_Acute_Care_Expected_Mortality, Admitting_Loc_GWU_PACU,
             Principal_Payer_NON_CONTRCTD_HMO_PPO, CDB_APR_DRG_All_Inpatient_Expected_Mortality,
             Admitting_Service_Code_NEU,
             Principal_Payer_Type_SELF_PAY_ADMIT, test)

trainsetrf1= setrf1[train_index,]
testsetrf1 = setrf1[-train_index,]
trainsetrf1$test=as.factor(trainsetrf1$test)
rf1= randomForest(test~.,data=trainsetrf1, ntree=500, type='classification')
print(rf1)
importance(rf1)
testsetrf1$test=as.factor(testsetrf1$test)
predictrf1=predict(rf1,testsetrf1)
table(testsetrf1$test)
table(predictrf1)
confusionMatrix(predictrf1, testsetrf1$test)
plot(rf1)
                      


