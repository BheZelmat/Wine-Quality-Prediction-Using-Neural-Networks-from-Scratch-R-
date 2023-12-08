library(reshape2)


# *********************************************************************
# ACTIVATION FUNCTIONS ************************************************
# *********************************************************************

# sigmoid

sigmoid <- function(x){
  
  1/(1+exp(-x))
  
}

# tanh- already known to R!

id <- function(x){x}

# make a "library" of activation functions

softmax <- function(z){
  
  exp(z) / sum(exp(z))
  
}

AF <- list(id, sigmoid, tanh, softmax)


# ***************************************************************************
# Derivatives of  activation functions 
# ***************************************************************************

Did <- function(x){1}

Dsig <- function(x){exp(-x) / ((1+exp(-x))^2)}

Dtanh <- function(x){1-(tanh(x)^2)}

D.softmax <- function(z){
  
  n <- length(z)
  S <- softmax(z)
  DS <- matrix(NA, nrow = n, ncol = n)
  
  for(i in 1:n){
    for(j in 1:n){
      if(i == j){
        D.i.j <- S[i]*(1-S[j])
      }else{
        D.i.j <- -S[i]*S[j]
      }
      DS[i,j] <- D.i.j
    }}
  return(DS)
}

DAF <- list(Did,Dsig,Dtanh, D.softmax)


# ******************************************************************************

initialize.parameters <- function(N,a){
  
  W1 <- matrix(runif(N[1]*N[2],-a,a),nrow = N[1])
  W2 <- matrix(runif(N[2]*N[3],-a,a),nrow = N[2])
  W3 <- matrix(runif(N[3]*N[4],-a,a),nrow = N[3])
  W4 <- matrix(runif(N[4]*N[5],-a,a),nrow = N[4])
  
  B1 <- matrix(runif(N[2],-a,a),nrow = N[2])
  B2 <- matrix(runif(N[3],-a,a),nrow = N[3])
  B3 <- matrix(runif(N[4],-a,a),nrow = N[4])
  B4 <- matrix(runif(N[5],-a,a),nrow = N[5])
  
  return(list("W" = list(W1,W2,W3,W4), "B" = list(B1,B2,B3,B4)))
  
}

# *********************************************************************
# LOSS FUNCTIONS ******************************************************
# *********************************************************************

mse <- function(A,K){
  
  return( mean((A-K)^2) )
  
}

D.mse <- function(A,K){
  
  return(2/length(A)*(A-K))
  
}

cross.entropy <- function(A, K){
  
  return(sum( K * log(1/A) ))
  
}

D.cross.entropy <- function(A, K){
  
  return(-K/A)
  
}

LF  <- list(mse, cross.entropy)
DLF <- list(D.mse, D.cross.entropy)

# ****************************************************************************

fwd.pass <- function(input, parameters, AF.chosen){
  
  W1 = parameters$W[[1]]
  W2 = parameters$W[[2]]
  W3 = parameters$W[[3]]
  W4 = parameters$W[[4]]

  B1 = parameters$B[[1]]
  B2 = parameters$B[[2]]
  B3 = parameters$B[[3]]
  B4 = parameters$B[[4]]
  
  AF1 = AF[[AF.chosen[1]]]
  AF2 = AF[[AF.chosen[2]]]
  AF3 = AF[[AF.chosen[3]]]
  AF4 = AF[[AF.chosen[4]]]
  AF5 = AF[[AF.chosen[5]]]
  
  Z1 = input
  A1 = AF1(Z1)
  Z2 = t(W1) %*% A1 + B1
  A2 = AF2(Z2)
  Z3 = t(W2) %*% A2 + B2
  A3 = AF3(Z3)
  Z4 = t(W3) %*% A3 + B3
  A4 = AF4(Z4)
  Z5 = t(W4) %*% A4 + B4
  A5 =AF5(Z5)
  
  return(list("Z" = list(Z1,Z2,Z3,Z4,Z5), "A" = list(A1,A2,A3,A4,A5)))
  
}

# *************************************************************************

back.propagation <- function(fwd.pass, 
                             parameters, 
                             AF.chosen, 
                             LF.chosen, 
                             known.input.result){
  
  # the loss function
  L  <- LF[[LF.chosen]]
  DL <- DLF[[LF.chosen]]
  
  # pre-active values
  Z5 <- fwd.pass$Z[[5]]
  Z4 <- fwd.pass$Z[[4]]
  Z3 <- fwd.pass$Z[[3]]
  Z2 <- fwd.pass$Z[[2]]
  Z1 <- fwd.pass$Z[[1]]
  
  # active values
  A5 <- fwd.pass$A[[5]]
  A4 <- fwd.pass$A[[4]]
  A3 <- fwd.pass$A[[3]]
  A2 <- fwd.pass$A[[2]]
  A1 <- fwd.pass$A[[1]]
  
  # net parameters
  W4 <- parameters$W[[4]]
  W3 <- parameters$W[[3]]
  W2 <- parameters$W[[2]]
  W1 <- parameters$W[[1]]
  
  
  B4 <- parameters$B[[4]]
  B3 <- parameters$B[[3]]
  B2 <- parameters$B[[2]]
  B1 <- parameters$B[[1]]
  
  # activation function derivatives
  D.AF5 <- DAF[[ AF.chosen[5] ]]
  D.AF4 <- DAF[[ AF.chosen[4] ]]
  D.AF3 <- DAF[[ AF.chosen[3] ]]
  D.AF2 <- DAF[[ AF.chosen[2] ]]
  
  # derivatives of loss w.r.t. neuron values Z:
  # if softmax is chosen:
  
  if(AF.chosen[[5]] == 4){
    DZ5 <- D.AF5( Z5 ) %*% DL(A5, known.input.result)
  }else{
    DZ5 <- DL(A5, known.input.result) * D.AF5( Z5 ) 
  }
  
  DZ4 <- (W4 %*% DZ5) * D.AF4( Z4 )
  DZ3 <- (W3 %*% DZ4) * D.AF3( Z3 )
  DZ2 <- (W2 %*% DZ3) * D.AF2( Z2 )
  
  # derivatives of loss w.r.t weights:
  DW4 <- A4 %*% t(DZ5)
  DW3 <- A3 %*% t(DZ4)
  DW2 <- A2 %*% t(DZ3)
  DW1 <- A1 %*% t(DZ2)
  
  # derivatives of loss w.r.t biases:
  DB4 <- DZ5
  DB3 <- DZ4
  DB2 <- DZ3
  DB1 <- DZ2
  
  return( list("DW" = list(DW1, DW2,DW3 ,DW4), "DB" = list(DB1, DB2, DB3, DB4) ) )
  
}

# *********************************************************************
# UPDATE PARAMETERS ***************************************************
# *********************************************************************

updateParameters <- function(parameters, gradients, step){
  
  DW4 <- gradients$DW[[4]]  
  DW3 <- gradients$DW[[3]]
  DW2 <- gradients$DW[[2]]  
  DW1 <- gradients$DW[[1]]
  
  DB4 <- gradients$DB[[4]]
  DB3 <- gradients$DB[[3]]  
  DB2 <- gradients$DB[[2]]
  DB1 <- gradients$DB[[1]]
  
  W4 <- parameters$W[[4]]
  W3 <- parameters$W[[3]]
  W2 <- parameters$W[[2]]
  W1 <- parameters$W[[1]]
  
  B4 <- parameters$B[[4]]
  B3 <- parameters$B[[3]]
  B2 <- parameters$B[[2]]
  B1 <- parameters$B[[1]]
  
  W1 <- W1 - step*DW1
  W2 <- W2 - step*DW2
  W3 <- W3 - step*DW3
  W4 <- W4 - step*DW4
  
  B4 <- B4 - step*DB4
  B3 <- B3 - step*DB3
  B2 <- B2 - step*DB2
  B1 <- B1 - step*DB1
  
  return(list("W"=list(W1, W2, W3, W4), "B"=list(B1, B2, B3, B4)))
  
}

# *********************************************************************
# ONE HOT *************************************************************
# *********************************************************************

one.hot.convert <- function(types, type){
  
  position <- which(types == type)
  oneHot   <- rep(0,length(types))
  oneHot[position] <- 1
  
  return(oneHot)
  
}

# *********************************************************************
# testNet *************************************************************
# *********************************************************************

testNet <- function(data){
  
  accuracy <- 0
  TP <- rep(0, numberOfTypes) # True Positives for each type
  FP <- rep(0, numberOfTypes) # False Positives for each type
  FN <- rep(0, numberOfTypes) # False Negatives for each type
  TN <- rep(0, numberOfTypes) # True Negatives for each type
  
  for(row in 1:nrow(data)){
    
    input      <- matrix(as.numeric(data[row, 1:M]), nrow = M)
    known      <- matrix(one.hot.convert(types, pull(data, classCol)[row]), nrow = numberOfTypes) 
    fPass      <- fwd.pass(input, parameters, AF.chosen)
    prediction <- round(fPass$A[[5]], 0)
    
    for(type in 1:numberOfTypes){
      if(known[type] == 1 && prediction[type] == 1){
        TP[type] <- TP[type] + 1
      } else if (known[type] == 0 && prediction[type] == 1){
        FP[type] <- FP[type] + 1
      } else if (known[type] == 1 && prediction[type] == 0){
        FN[type] <- FN[type] + 1
      } else {
        TN[type] <- TN[type] + 1
      }
    }
    
    if(all(known == prediction)){
      accuracy <- accuracy + 1
    }
  }
  
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1 <- 2 * (precision * recall) / (precision + recall)
  
  overall_precision <- mean(precision, na.rm = TRUE)
  overall_recall <- mean(recall, na.rm = TRUE)
  overall_F1 <- mean(F1, na.rm = TRUE)
  
  return(list(accuracy = accuracy / nrow(data) * 100,
              precision = overall_precision * 100,
              recall = overall_recall* 100,
              F1 = overall_F1* 100))
}
############################################################################
### Confusion Matrix ######################################################
###########################################################################




plotClassificationMatrix <- function(data, title = "Classification Matrix") {
  actualValues <- NULL
  predictedValues <- NULL
  
  for(row in 1:nrow(data)){
    input <- matrix(as.numeric(data[row, 1:M]), nrow = M)
    actual <- matrix(one.hot.convert(types, pull(data, classCol)[row]), nrow = numberOfTypes) 
    fPass <- fwd.pass(input, parameters, AF.chosen)
    predicted <- round(fPass$A[[5]], 0)
    
    actualClass <- which.max(actual)
    predictedClass <- which.max(predicted)
    
    actualValues <- c(actualValues, actualClass)
    predictedValues <- c(predictedValues, predictedClass)
  }
  
  confusionMatrix <- table(Actual = actualValues, Predicted = predictedValues)
  matrix_df <- as.data.frame(confusionMatrix)
  matrix_melt <- melt(matrix_df, id.vars = c("Actual", "Predicted"))
  
  ggplot(matrix_melt, aes(x = Actual, y = Predicted, fill = factor(value))) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%d", value)), vjust = 1) +
    scale_fill_manual(values = rainbow(length(unique(matrix_melt$value)))) +
    labs(title = title, x = "Actual Label", y = "Predicted Label") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}
