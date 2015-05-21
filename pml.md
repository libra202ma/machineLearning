# Machine Learning Model Building for Activity Tracking Device Data
Wednesday, May 20, 2015  
###Synopsis
This report applies the practical machine learning algorithm to fit the activity tracking device data of a group of people doing barbell lifts correctly and incorrectly in 5 different ways, builds a model in order to predict the manner in which they did the exercise.

###Data Processing
Load required packages and data:

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
data <- read.csv("pml-training.csv")
```
Un-select first 6 columns of index, username and timestamp:

```r
data <- data[,7:160]
```
Un-select columns with mostly missing values.

```r
data[data=='']<-NA
naa <- apply(data,2,function(x) mean(is.na(x)))
data_clean <- data[,naa<0.5]
```

###Model Building
Split data into training set and testing set:

```r
InTrain <- createDataPartition(data_clean$classe,p=0.75,list=FALSE)
training <- data_clean[InTrain,]
testing <- data_clean[-InTrain,]
```

Set fit control parameters: using 5-fold cross validation

```r
fitControl <- trainControl(method="cv",number=5,repeats=2,allowParallel=TRUE)
tgrid <- expand.grid(mtry=c(6))
```
Fit a boosting tree model via gbm package:

```r
set.seed(565)
rfFit1 <- train(classe~., data=training,method='rf',trControl=fitControl,tuneGrid = tgrid, verbose=TRUE)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
rfFit1
```

```
## Random Forest 
## 
## 14718 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 11774, 11775, 11774, 11774, 11775 
## 
## Resampling results
## 
##   Accuracy  Kappa      Accuracy SD  Kappa SD   
##   0.996535  0.9956171  0.001755013  0.002219994
## 
## Tuning parameter 'mtry' was held constant at a value of 6
## 
```

```r
print(rfFit1$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, verbose = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 6
## 
##         OOB estimate of  error rate: 0.35%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4185    0    0    0    0 0.000000000
## B    5 2840    3    0    0 0.002808989
## C    0   12 2555    0    0 0.004674718
## D    0    0   23 2388    1 0.009950249
## E    0    0    0    8 2698 0.002956393
```
The out of sample error estimated by the cross-validation is <0.5%, which is very low. 

###Model Testing
Do confusionMatrix on the testing data set for out of sample error:

```r
cm <- confusionMatrix(testing$classe,predict(rfFit1,testing))
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    0    0    0    1
##          B    3  946    0    0    0
##          C    0    1  854    0    0
##          D    0    0    9  795    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9971          
##                  95% CI : (0.9952, 0.9984)
##     No Information Rate : 0.2849          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9964          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9979   0.9989   0.9896   1.0000   0.9989
## Specificity            0.9997   0.9992   0.9998   0.9978   1.0000
## Pos Pred Value         0.9993   0.9968   0.9988   0.9888   1.0000
## Neg Pred Value         0.9991   0.9997   0.9978   1.0000   0.9998
## Prevalence             0.2849   0.1931   0.1760   0.1621   0.1839
## Detection Rate         0.2843   0.1929   0.1741   0.1621   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9988   0.9991   0.9947   0.9989   0.9994
```
The accuracy is >0.99, which is very high accuracy.
