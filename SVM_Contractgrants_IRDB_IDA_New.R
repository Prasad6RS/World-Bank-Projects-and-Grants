#Setting the working directory and fetching
setwd("C:\\NCI - Studies\\_DMML\\Datasets for Project")
getwd()

#Reading the input files : IBRD loan statements, IDA credits and grants, and contract grants
cGrants<-read.csv("Contract_grants_IRDB_IDA.csv")
irdbLoan<-read.csv("IBRD_Statement_Of_Loans_-_Historical_Data.csv")
idaLoan<-read.csv("IDA_Statement_Of_Credits_and_Grants_-_Historical_Data.csv")

str(cGrants)

#Importing required libraries
library(dplyr)
library(tidyverse)
library(ggplot2)
library(Matrix)
library(caret)
library(caTools)
library(e1071)
library(kernlab)
library(ISLR)
library(RColorBrewer)
library(pROC)

#Sampling: First 50,000 rows
#cGrant<-cGrants
#cGrants<-cGrant[1:50000,]
#str(cGrants)

#Appending predictor variable columns from IRDB Loan Statement
match(cGrants$Project.ID, irdbLoan$Project.ID)
irdbLoan$Disbursed.Amount[match(cGrants$Project.ID, irdbLoan$Project.ID)]
cGrants$irdbAmount<-irdbLoan$Disbursed.Amount[match(cGrants$Project.ID, irdbLoan$Project.ID)]
str(cGrants)

#Replacing the values less than or equal to 0 to 1; 1 is fairly small when compared to the loans amounts averaging millions;
#Replacement is necessary because logarithm would be applied next
cGrants$irdbAmount[cGrants$irdbAmount<=0]<-1

#Appending predictor variable columns from IDA credits and grants
cGrants$idaAmount<-idaLoan$Disbursed.Amount[match(cGrants$Project.ID, idaLoan$Project.ID)]
str(cGrants)

#Omitting the NA values
na.omit(cGrants)

#Replacing the values less than or equal to 0 to 1; 1 is fairly small when compared to the loans amounts averaging millions;
#Replacement is necessary because logarithm would be applied next
cGrants$idaAmount[cGrants$idaAmount<=0]<-1


#Removing columns that have NA, and columns not used in the analysis
colSums(is.na(cGrants))
cGrants<-cGrants[,!(names(cGrants)%in% c("As.of.Date","Fiscal.Year","Region","Borrower.Country",
                                         "Borrower.Country.Code","Project.ID","Project.Name",
                                         "Procurement.Type","Procurement.Category","Procurement.Method",
                                         "Product.line","WB.Contract.Number","Contract.Description",
                                         "Contract.Signing.Date","Supplier","Supplier.Country",
                                         "Supplier.Country.Code","Supplier.State",
                                         "Borrower.Contract.Reference.Number"))]
#Display the structure
str(cGrants)

#Converting the predictor, Major Sector, to factor
cGrants$Major.Sector<-as.factor(cGrants$Major.Sector)
str(cGrants)

#Applying logarithm to variables with large values to standardize
cGrants$Total.Contract.Amount..USD.<-log(cGrants$Total.Contract.Amount..USD.)
cGrants$irdbAmount<-log(cGrants$irdbAmount)
cGrants$idaAmount<-log(cGrants$idaAmount)

#Sampling: Test and Train Partitioning at 70%
split_data<-sample.split(cGrants$Total.Contract.Amount..USD.,SplitRatio = 0.7)
train<-subset(cGrants,split_data==T)
test<-subset(cGrants,split_data==F)
str(train)

#Applying machine learning models

#######################################################################################################################

#Modeling  using SVM technique using radial kernal

#######################################################################################################################

mod1<-svm(Total.Contract.Amount..USD.~irdbAmount+idaAmount,
          data=train, type="nu-regression", kernal="radial", shrinking=T, gamma = 0.3)

#Prediction
cGrantPredict1<-predict(mod1,test)


#RMSE
RMSE(test$Total.Contract.Amount..USD.,cGrantPredict1)
range(cGrantPredict1)

#######################################################################################################################

#Modeling  using SVM technique using linear kernal

#######################################################################################################################

mod2<-svm(Total.Contract.Amount..USD.~irdbAmount+idaAmount,
          data=train, type="nu-regression", kernal="linear", shrinking=T, gamma = 0.4)

#Scatter plot to see the predict vs actual values
plot(test$Total.Contract.Amount..USD., pch=4, main = "Scatter Plot")
points(test$Total.Contract.Amount..USD., cGrantPredict1, col = "red", pch=4)

#Prediction
cGrantPredict2<-predict(mod2,test)

#RMSE
RMSE(test$Total.Contract.Amount..USD.,cGrantPredict2)
range(cGrantPredict2)


#######################################################################################################################

#Modeling  using SVM technique using polynomial kernal

#######################################################################################################################

mod3<-svm(Total.Contract.Amount..USD.~irdbAmount+idaAmount,data=train,
          type="nu-regression", kernal="polynomial", shrinking=T, gamma = 0.1)

#Prediction
cGrantPredict3<-predict(mod3,test)

#RMSE
RMSE(test$Total.Contract.Amount..USD.,cGrantPredict3)
range(cGrantPredict3)

#######################################################################################################################

#Modeling  using SVM technique using sigmoid kernal

#######################################################################################################################

mod4<-svm(Total.Contract.Amount..USD.~irdbAmount+idaAmount,data=train,
          type="nu-regression", kernal="sigmoid", shrinking=T, gamma = 0.2)

#Prediction
cGrantPredict4<-predict(mod4,test)

#RMSE
RMSE(test$Total.Contract.Amount..USD.,cGrantPredict4)
range(cGrantPredict4)