
# Abstract  :- 
# Human Activity Recognition - HAR - has emerged as a key research area in the last years 
# and is gaining increasing attention by the pervasive computing research community, 
# especially for the development of context-aware systems.
# Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect 
# a large amount of data about personal activity relatively inexpensively. 
# These type of devices are part of the quantified self movement 
# - a group of enthusiasts who take measurements about themselves regularly to 
# improve their health, to find patterns in their behavior, or because they are tech geeks. 
# One thing that people regularly do is quantify how much of a particular activity they do,
# but they rarely quantify how well they do it. In this project, your goal will be to use data 
# from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 
# They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

# Project Goals:-
  
#  -The goal of your project is to predict the manner in which the exercise was done.

#vA full description is available at the site where the data was obtained:
  
  
#  <http://groupware.les.inf.puc-rio.br/har>

#Loading all required packages

require(caret) 
require(kernlab)
require(plyr)
require(dplyr)
require(car)
require(MASS)
require(ggplot2)
require(corrgram)
require(MASS)
require(reshape)
require(scales)


## Load and Clean the data , as many observation in dataset contains NAs and blank 

dataM1 <- read.csv("pml-training.csv",stringsAsFactors=FALSE)
str(dataM1)
head(dataM1)
dim(dataM1)


remV1 <- as.matrix(apply(dataM1, 2, function(x) length(which(is.na(x)))))
remVn1 <- print(names(remV1[which(remV1[,1]==0),]))
myvars1 <- names(dataM1) %in% remVn1
dataM2 <- dataM1[,myvars1]
summary(dataM2)

remV2 <- as.matrix(apply(dataM2, 2, function(x) length(which(x==""))))
remVn2 <- print(names(remV2[which(remV2[,1]==0),]))
myvars2 <- names(dataM2) %in% remVn2
dataM3 <- dataM2[,myvars2]
summary(dataM3)

## First six preditors has no effect on our prediction goals and hence are being removed 
dataM3 <- dataM3[,-c(1,2,3,4,5,6)]

## data partition for training and validation, the split is 75% for training

set.seed(1234)
inTrain <- createDataPartition(y=dataM3$classe,p=0.75, list=FALSE)
training1 <- dataM3[inTrain,]
testing1 <- dataM3[-inTrain,]
dim(training1)

## centering and scaling to normalise all the predictors,
## applied to validation dataset as well

preProcValues <- preProcess(dataM3[,-54], method = c("center", "scale"))

trainTransformed <- predict(preProcValues, training1[,-54])
testTransformed <- predict(preProcValues, testing1[,-54])

trainTransformedf <- cbind(trainTransformed,training1[,54])
testTransformedf <-  cbind(testTransformed,testing1[,54])
names(trainTransformedf)[54] <- c("classe")
names(testTransformedf)[54] <- c("classe")

# training control parameters applied to train the model using caret,method = "repeatedcv", number = 3,repeats = 1

set.seed(32323)  
folds <-  createFolds(y=trainTransformedf[,54],k=10,list=TRUE,returnTrain=TRUE)
fitControl1 <- trainControl(method = "repeatedcv", number = 3,repeats = 1)

# random forest
# 
# rfH <- train(classe ~ .,data=trainTransformedf[,2:54],
#              method = "rf",importance=TRUE,
#              prox=TRUE,
#              trControl = fitControl1)

print(rfH)
plot(rfH)

# Visual inspection of the importance of variables, created using random forest 
dm <- as.matrix(varImp(rfH$finalMode))
dm1 <- dm[1:52,]
dm2 <- melt(dm1, id=c("row.names")) 
dm3 <- ddply(dm2, .(X1), transform, scale = rescale(value))
p <- ggplot(dm3, aes(X2,X1)) + geom_tile(aes(fill = scale)) 
p <- p+ggtitle("Variable Importance in Prediction")
p <- p+ylab("Variables") +xlab("Class")
p


# neural networks

nnetH <- train(classe ~ .,data=trainTransformedf[,2:54],
               method = "nnet",tuneLength = 2,
               trControl = fitControl,linout=FALSE,trace=FALSE)

# knn 
knnH <- train(classe ~ .,data=trainTransformedf[,2:54],
              method = "knn",
              tuneLength = 2,
              trControl = fitControl)


# Support Vector machine

svmH <- train(classe ~ .,data=trainTransformedf[,2:54], method = "svmRadialCost", trControl = fitControl)




prednnetHTr <- predict(nnetH,trainTransformedf[,2:53])
prednnetHTe <- predict(nnetH,testTransformedf[,2:53])


predknnHTr <- predict(knnH,trainTransformedf[,2:53])
predknnHTe <- predict(knnH,testTransformedf[,2:53])

predsvrHTr <- predict(svmH,trainTransformedf[,2:53])
predsvrHTe <- predict(svmH,testTransformedf[,2:53])

predrfHTr <- predict(rfH,trainTransformedf[,2:53])
predrfHTe <- predict(rfH,testTransformedf[,2:53])

# combining model based on majority voting
preDFH <- data.frame(prednnetHTe,predknnHTe,predsvrHTe,predrfHTe)
df1 <- data.frame()
df3 <- data.frame()
for (i in 1:nrow(preDFH)) {
  a <- preDFH[i,]=='A'
  b <- preDFH[i,]=='B'
  c <- preDFH[i,]=='C'
  d <- preDFH[i,]=='D'
  d <- preDFH[i,]=='D'
  e <- preDFH[i,]=='E'
  df <- data.frame(length(which(a)),length(which(b)),length(which(c)),length(which(d)),length(which(e)))
  df1 <- rbind(df1,df)
}

df3 <- apply(df1,1,function(x) which.max(x))
df3[df3==1]<-'A'
df3[df3==2]<-'B'
df3[df3==3]<-'C'
df3[df3==4]<-'D'
df3[df3==5]<-'E'
ensmbStackH <- data.frame(df3)

# Comparison between insample and out sample accuracy 
confusionMatrix(trainTransformedf[,54],prednnetHTr)
confusionMatrix(testTransformedf[,54],prednnetHTe)

confusionMatrix(trainTransformedf[,54],predknnHTr)
confusionMatrix(testTransformedf[,54],predknnHTe)


confusionMatrix(trainTransformedf[,54],predsvrHTr)
confusionMatrix(testTransformedf[,54],predsvrHTe)

confusionMatrix(trainTransformedf[,54],predrfHTr)
confusionMatrix(testTransformedf[,54],predrfHTe)

# Comparison between models accuracy 
confusionMatrix(testTransformedf[,54],prednnetHTe)
confusionMatrix(testTransformedf[,54],predknnHTe)
confusionMatrix(testTransformedf[,54],predsvrHTe)
confusionMatrix(testTransformedf[,54],predrfHTe)
confusionMatrix(testTransformedf[,54],ensmbStackH[,1])



# As random forest provides maximum accuracy in prediction across all classes, hence 
# it was chosen to predict the classes in test cases

# The out of sample accuracy for Randon Forest model is 

ouSampAcc <- confusionMatrix(testTransformedf[,54],predrfHTe)

# The out of sample error for Randon Forest model is :-
ouSamperr <- paste(round((1-ouSampAcc$overall[1])*100,2),'%')


## Prediciting Cases for submission 
datat1 <- read.csv("pml-testing.csv",stringsAsFactors=FALSE)

remVt1 <- as.matrix(apply(datat1, 2, function(x) length(which(is.na(x)))))
remVnt1 <- print(names(remVt1[which(remVt1[,1]==0),]))
myvarst1 <- names(datat1) %in% remVnt1
datat2 <- datat1[,myvarst1]


remVt2 <- as.matrix(apply(datat2, 2, function(x) length(which(x==""))))
remVnt2 <- print(names(remVt2[which(remVt2[,1]==0),]))
myvarst2 <- names(datat2) %in% remVnt2
datat3 <- datat2[,myvarst2]
datat3 <- datat3[,-c(1,2,3,4,5,6)]


finaltest <- predict(preProcValues, datat3[,-54])
finaltestf <-  cbind(finaltest,datat3[,54])


predrfHTeff <- predict(rfH,finaltestf[,2:53])
predrfHTef <- data.frame(finaltestf[,54],predrfHTeff)
names(predrfHTef) <- c("problem_id","answers")
ensmbStackHfa <- as.character(predrfHTef[,2])
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(ensmbStackHfa)
print(ensmbStackHfa)