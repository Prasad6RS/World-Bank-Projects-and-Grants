#import the libraries required
library(tidyverse)
library(dplyr)
library(mlr)
library(xgboost)
library(Metrics)
library(knitr)
library(pROC)
library(caTools)
library(caret)

#libraries for Visualization
library(ggplot2)
library(ggthemes)

#Setting the working directory and fetching
setwd("C:\\NCI - Studies\\_DMML\\Datasets for Project")
getwd()

#Reading the input files : IBRD loan statements, IDA credits and grants, and contract grants
cGrants<-read.csv("Contract_grants_IRDB_IDA.csv")
irdbLoan<-read.csv("IBRD_Statement_Of_Loans_-_Historical_Data.csv")
idaLoan<-read.csv("IDA_Statement_Of_Credits_and_Grants_-_Historical_Data.csv")


#Structure of Contract_Grants
str(cGrants)

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
cGrants<-na.omit(cGrants)

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

#Structure of Contract_Grants
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

#Convert data.frame/tibble to DMatrix
traindata_x <- train %>% select(irdbAmount,idaAmount)
traindata_y <- train$Total.Contract.Amount..USD.

testdata_x <- test %>% select(irdbAmount,idaAmount)

dtrain <- xgb.DMatrix(data=data.matrix(traindata_x), label=traindata_y)
dtest <- xgb.DMatrix(data=data.matrix(testdata_x))
watchlist <- list(traindata=dtrain)


#Setting up Hyperparamters for XGBoost
#Set nfolds = 3, nrounds = 100
nfolds <- 3
nrounds <- 100

params <- list("max_depth"=3,
               "booster" = "gbtree",
               "colsample_bytree"=0.3,
               "min_child_weight"=1,
               "subsample"=0.8,
               "eval_metric"= "rmse", 
               "objective"= "reg:squarederror")


#Applying XG Boost Model
model_xgb <- xgb.train(params=params,
                       data=dtrain,
                       nrounds=nrounds,
                       maximize=FALSE,
                       watchlist=watchlist,
                       print_every_n=3)
summary(model_xgb)
model_xgb

#predicting the model
predicted <- predict(model_xgb, dtest)
summary(predicted)
head(predicted)

#RMSE
RMSE(test$Total.Contract.Amount..USD.,predicted)
range(predicted)