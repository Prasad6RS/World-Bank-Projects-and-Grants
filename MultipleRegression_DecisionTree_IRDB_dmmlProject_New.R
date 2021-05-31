#Setting the working directory and fetching
setwd("C:\\NCI - Studies\\_DMML\\Datasets for Project")
getwd()

#Reading the input files : IDA credits and grants
irdbLoan<-read.csv("IBRD_Statement_Of_Loans_-_Historical_Data.csv")
str(irdbLoan)

#Importing required libraries
library(dplyr)
library(tidyverse)
library(ggplot2)
library(Matrix)
library(caret)
library(caTools)
library(rpart)
library(rpart.plot)
library(tree)
library(rattle)
library(pROC)
library(corrplot)
library(car)

#Sampling: First 50,000 rows
irdb<-irdbLoan
sirdb<-irdb[1:50000,]
str(sirdb)

#Removing columns that have NA, and columns not used in the analysis
colSums(is.na(sirdb))
sirdb<-sirdb[,!(names(sirdb)%in% c("End.of.Period","Loan.Number",
                                   "Region","Country.Code","Country","Borrower",
                                   "Guarantor.Country.Code","Guarantor",
                                   "Currency.of.Commitment","Project.ID",
                                   "Project.Name","Exchange.Adjustment",
                                   "Borrower.s.Obligation","First.Repayment.Date",
                                   "Last.Repayment.Date","Agreement.Signing.Date",
                                   "Board.Approval.Date","Effective.Date..Most.Recent.",
                                   "Closed.Date..Most.Recent.","Last.Disbursement.Date",
                                   "Currency.of.Commitment"))]

#Converting the predictors, Loan Type and Loan Status, to factor
sirdb$Loan.Type<-as.factor(sirdb$Loan.Type)
sirdb$Loan.Status<-as.factor(sirdb$Loan.Status)
str(sirdb)

#Transforming NA in InterestRate column to mean of InterestRate and omitting NA in LoansHeld
sirdb$Interest.Rate<-ifelse(is.na(sirdb$Interest.Rate),
                            mean(sirdb$Interest.Rate, na.rm = T),sirdb$Interest.Rate)
sirdb$Loans.Held[is.na(sirdb$Loans.Held)]=1

#Replacing the values less than or equal to 0 to 1.1; 1.1 is fairly small when compared to the loans amounts averaging millions;
#Replacement is necessary because logarithm would be applied next
sirdb$Repaid.to.IBRD[sirdb$Repaid.to.IBRD==0]<-1.1
sirdb$Disbursed.Amount[sirdb$Disbursed.Amount==0]<-1.1
sirdb$Loans.Held[sirdb$Loans.Held<=1]<-1.1
sirdb$Original.Principal.Amount[sirdb$Original.Principal.Amount==0]<-1.1
#replace(sirdb$Repaid.to.IBRD,sirdb$Repaid.to.IBRD < 1,1)
str(sirdb)
#sirdb<-na.omit(sirdb)
colSums(is.na(sirdb))

#Applying logarithm to variables with large values to standardize
sirdb$Disbursed.Amount<-log(sirdb$Disbursed.Amount)
sirdb$Repaid.to.IBRD<-log(sirdb$Repaid.to.IBRD)
sirdb$Original.Principal.Amount<-log(sirdb$Original.Principal.Amount)
#sirdb$Loans.Held<-log(sirdb$Loans.Held)
colSums(is.na(sirdb))

#to determine the correlation plot
sirdb1<-sirdb
sirdb1<-sirdb1[,!(names(sirdb1)%in% c("Cancelled.Amount",
                                      "Undisbursed.Amount","Due.to.IBRD","Sold.3rd.Party","Repaid.3rd.Party",
                                      "Due.3rd.Party"))]
sirdb1$Loan.Type<-as.numeric(sirdb1$Loan.Type)
sirdb1$Loan.Status<-as.numeric(sirdb1$Loan.Status)
relation<-cor(sirdb1)
corrplot(relation,method = "square", order = "AOE",
         tl.col = "Black", tl.srt = 20)


#Sampling: Test and Train Partitioning at 70%
my_data<-sample.split(sirdb$Disbursed.Amount,SplitRatio = 0.70)
train<-subset(sirdb,my_data==T)
test<-subset(sirdb,my_data==F)

nrow(train)
nrow(test)


#Applying machine learning models

#######################################################################################################################

#Modeling using multiple linear regression

#######################################################################################################################

linRegression<-lm(Disbursed.Amount~Loan.Status+Interest.Rate+
                   Repaid.to.IBRD+Original.Principal.Amount, data = train)

myPredict<-predict(linRegression, newdata = test)

#Variance Inflation Factor
vif(linRegression)

#Durbin Watson test to check for auto-correlation among residuals
durbinWatsonTest(linRegression)

#RMSE
RMSE(test$Disbursed.Amount,myPredict)



#######################################################################################################################

#Modeling using decision trees

#######################################################################################################################

dTree<-rpart(Disbursed.Amount~Loan.Status+Interest.Rate+
               Repaid.to.IBRD+Original.Principal.Amount, data = train)

dPredict<-predict(dTree, newdata = test)
fancyRpartPlot(dTree)


#RMSE
RMSE(test$Disbursed.Amount,dPredict)
range(myPredict)


#Explaing residuals
par(mfrow = c(2,2))
plot(linRegression)