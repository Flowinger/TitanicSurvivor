library(dplyr)
library(randomForest)
library(caret)
library(rpart)

train = read.csv("train_titanic.csv")
test = read.csv("test_titanic.csv")

head(train)

transform_df <- function(dataframe) {
  # Replace missing values in 'Age' with average age
  dataframe$Age <- replace(dataframe$Age,is.na(dataframe$Age),mean(dataframe$Age,na.rm=TRUE))
  dataframe$Fare <- replace(dataframe$Fare,is.na(dataframe$Fare),mean(dataframe$Fare,na.rm=TRUE))
  dataframe$Embarked <- replace(dataframe$Embarked,is.na(dataframe$Embarked),"S")
  # One-Hot vector encoding
  dataframe <- dataframe %>% mutate(Sex = ifelse(Sex == 'male',0,1)) 
  #dataframe <- dataframe %>% mutate(Cabin = ifelse(Cabin == '',0,1))
  #dataframe <- dataframe %>% mutate(embarked_s = ifelse(Embarked != 'S',0,1))
  #
  #dataframe <- dataframe %>% mutate(embarked_c = ifelse(Embarked != 'C',0,1))
  #dataframe <- dataframe %>% mutate(pclass_1 = ifelse(Pclass != 1,0,1))
  #dataframe <- dataframe %>% mutate(pclass_2 = ifelse(Pclass != 2,0,1))
  #dataframe <- dataframe %>% mutate(pclass_3 = ifelse(Pclass != 3,0,1))
  #dataframe <- dataframe %>% mutate(Pclass = ifelse(Pclass <= 2.5,0,1))
  #dataframe <- dataframe %>% mutate(adult = ifelse(Age < 18,0,1))
  
  # Title
  dataframe$Title <- gsub('(.*, )|(\\..*)', '', dataframe$Name)
  officer <- c('Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev')
  royalty <- c('Dona', 'Lady', 'the Countess','Sir', 'Jonkheer')
  dataframe$Title[dataframe$Title == 'Mlle']        <- 'Miss' 
  dataframe$Title[dataframe$Title == 'Ms']          <- 'Miss'
  dataframe$Title[dataframe$Title == 'Mme']         <- 'Mrs' 
  dataframe$Title[dataframe$Title %in% royalty]  <- 'Royalty'
  dataframe$Title[dataframe$Title %in% officer]  <- 'Officer'
  dataframe$Title <- factor(dataframe$Title)
  
  #dataframe$Title[dataframe$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
  #dataframe$Title[dataframe$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
  #dataframe$Title[dataframe$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
  # Convert to a factor
  #dataframe$Title <- factor(dataframe$Title)
  
  #dataframe <- dataframe %>% mutate(Fare = ifelse(Fare <= 50,0,1))
  dataframe$FamilySize <- dataframe$SibSp+dataframe$Parch+1
  #dataframe <- dataframe %>% mutate(FamilyID = ifelse(FamilySize <= 3,0,1))
  #dataframe$farePerPerson <- dataframe$Fare/(dataframe$SibSp+dataframe$Parch+1)
  # Name
  dataframe$Name <- as.character(dataframe$Name)
  dataframe$Surname <- sapply(dataframe$Name, FUN=function(x) {strsplit(x, split="[,.]")[[1]][1]})
  dataframe$FamilyID2 <- paste(as.character(dataframe$FamilySize), dataframe$Surname, sep="")
  dataframe$FamilyID2[dataframe$FamilySize <= 2] <- 'Small'
  dataframe$FamilyID2 <- factor(dataframe$FamilyID2)
  
  dataframe$Pclass <- factor(dataframe$Pclass)
  dataframe$Embarked <- factor(dataframe$Embarked)
  # Delete columns
  dataframe <- select(dataframe, -c(Ticket,Name,Cabin,PassengerId,Surname))
  #return(dataframe)
  dataframe
}

train_set = transform_df(train)
train_set$Survived <- as.factor(train_set$Survived)
test_set = transform_df(test)

#test_index = createDataPartition(dataframe$Survived, p=0.75, list=FALSE)
#test_set <- dataframe[-test_index,]
#train_set <- dataframe[test_index,]

# People survived
percentage <- prop.table(table(train_set$Survived)) * 100
cbind(freq=table(train_set$Survived), percentage=percentage)

summary(train_set)

X <- train_set[,2:9]
y <- train_set[,1]

#train_set$Age <- scale(train_set$Age)
#test_set$Age <- scale(test_set$Age)
#train_set$Fare <- scale(train_set$Fare)
#test_set$Fare <- scale(test_set$Fare)
#train_set$farePerPerson <- scale(train_set$farePerPerson)
#test_set$farePerPerson <- scale(test_set$farePerPerson)


# look at distributions of features
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=X,y=y,plot="density",scales=scales)

# Cross-val 10 fold
cv <- trainControl(method = "cv", number=10,sampling = "up")
# CART, RF, SVM
set.seed(7)
fit.cart <- train(Survived~.,data = train_set,method="rpart",metric="Accuracy",trControl=cv)
set.seed(7)
fit.rf <- train(Survived~.,data = train_set,method="rf",metric="Accuracy",trControl=cv,cutoff=c(0.4,0.6))
set.seed(7)
fit.svm <- train(Survived~.,data = train_set,method="svmRadial",metric="Accuracy",trControl=cv,cutoff=c(0.6,0.4))
set.seed(7)
fit.gbm <- train(Survived~.,data = train_set,method="gbm",metric="Accuracy",trControl=cv,verbose=FALSE)
set.seed(415)
fit.cf <- cforest(Survived~., data = train_set,controls=cforest_unbiased(ntree=2000, mtry=3))

results <- resamples(list(cart=fit.cart,rf=fit.rf,svm=fit.svm,gbm=fit.gbm))
summary(results)

print(fit.svm)

dotplot(results)

# Predictions
preds = predict(fit.svm, test_set)
preds_rf = predict(fit.rf, test_set)
preds_gbm = predict(fit.gbm, test_set)
preds_cf = predict(fit.cf, test_set,OOB=TRUE,type="response")

# Confusion Matrix
sub = read.csv("gender_submission.csv")
confusionMatrix(preds,sub$Survived)
confusionMatrix(preds_rf,sub$Survived)
confusionMatrix(preds_gbm,sub$Survived)
confusionMatrix(preds_cf,sub$Survived)

# SVM
submission = read.csv("gender_submission.csv")
submission$Survived = preds
write.csv(submission,file="submission_SVM7.csv",row.names = F)
# GradientBoosting
submission_gbm = read.csv("gender_submission.csv")
submission_gbm$Survived = preds_gbm
write.csv(submission_gbm,file="submission_GBM-1.csv",row.names = F)
# RandomForest
submission_rf = read.csv("gender_submission.csv")
submission_rf$Survived = preds_rf
write.csv(submission_rf,file="submission_RF25.csv",row.names = F)
# Conditional RF
submission_cf = read.csv("gender_submission.csv")
submission_cf$Survived = preds_cf
write.csv(submission_cf,file="submission_CF2.csv",row.names = F)


# Take a look at feature importances
library(party)
varImp(fit.cart)

RF_fit = randomForest(Survived~.,data = train_set,cutoff=c(0.57,0.43),ntree=5000,trControl=cv)
RF_pred = predict(RF_fit, test_set)
confusionMatrix(RF_pred,sub$Survived)

# RF .9067 with scaling
submission_RF = read.csv("gender_submission.csv")
submission_RF$Survived = RF_pred
write.csv(submission,file="submission_RF4.csv",row.names = F)

# SVM 0.8947
library(e1071)
svm_model = svm(Survived~.,data = train_set,cutoff=c(0.6,0.4),trControl=cv)
svm_pred = predict(svm_model,test_set)
confusionMatrix(svm_pred,sub$Survived)

submission_SVM = read.csv("gender_submission.csv")
submission_SVM$Survived = svm_pred
write.csv(submission,file="submission_SVM-1.csv",row.names = F)

