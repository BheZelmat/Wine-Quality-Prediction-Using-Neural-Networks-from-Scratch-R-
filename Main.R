# *************************************************************************************************
# Packages & Functions ----------------------------------------------------------------------------
# *************************************************************************************************

source("NN_functions.R")

library(tidyverse)




# *************************************************************************************************
# Necessary functions  ------------------------------------------------------------------
# *************************************************************************************************


# scaling function
rescale <- function(x, to = c(0, 1)) {
  # Extract the min and max of the new scale
  min_to <- min(to)
  max_to <- max(to)
  
  # Scale the data
  x_std <- (x - min(x)) / (max(x) - min(x))
  x_scaled <- x_std * (max_to - min_to) + min_to
  
  return(x_scaled)
}


# *************************************************************************************************
# Load Data Set ------------------------------------------------------------------
# *************************************************************************************************

data <- read.csv("wine.csv")

dim(data)

View(data)

index.classification.column<-ncol(data)
# scale the data
type.of.wine       <- data[,index.classification.column]
data.temp          <- data[,-index.classification.column]
data.temp          <- apply(data.temp,2,rescale,to=c(0,1))
data               <- cbind.data.frame(data.temp, type.of.wine)

types.wine <- sort(  unique(  pull(  data, index.classification.column  )  )  )
types.wine

# rename classification column to "class"
names(data)[ncol(data)] = "class"

View(data)
# train / test split (80%/20%)

index1   <- sample(1:nrow(data), size = 0.2*nrow(data))
training <- data[-index1,]
testing  <- data[index1,]

# randomize the order of training:

training <- training[sample(1:nrow(training), nrow(training)),]

# view training & testing distributions

ggplot(stack(training[,1:(ncol(training)-1)]), aes(x = ind, y = values)) +
  geom_boxplot() +
  labs(title = "Distributions of Training set",
       x = "Features",
       y = "Distribution",
       caption = "Caption Text") 


ggplot(stack(testing[,1:(ncol(testing)-1)]), aes(x = ind, y = values)) +
  geom_boxplot()+
  labs(title = "Distributions of Testing set",
       x = "Features",
       y = "Distribution",
       caption = "Caption Text") 


# *************************************************************************************************
# Neural Net Setup --------------------------------------------------------------------------------
# *************************************************************************************************

classCol            <- ncol(data)
types               <- sort(unique(pull(data,classCol)))
M                   <- classCol - 1
numberOfTypes       <- length( unique( pull(data,classCol) ) ) 
types               <- sort(unique(pull(data, classCol)))
N                   <- c(M,2*M,2*M,2*M,numberOfTypes)
AF.chosen           <- c(1,3,2,2,4)
lossFunctionChosen  <- 2
loss                <- LF[[lossFunctionChosen]]
step                <- 0.01
epochs              <- 100

# *************************************************************************************************
# RUN THE NET AND PLOT THE LOSS *******************************************************************
# *************************************************************************************************






parameters <- initialize.parameters(N,0.5)
es<-c()
epochlosses<-c()
for(e in 1:epochs){
  
  losses <- c()
  
  for(row in 1:nrow(training) ){
    
    input      <- matrix(as.numeric(training[row,1:M]), nrow = M)
    known      <- matrix(one.hot.convert(types, pull(training, classCol)[row] ), nrow = numberOfTypes) 
    fPass      <- fwd.pass(input, parameters, AF.chosen)
    grads      <- back.propagation(fPass, parameters, AF.chosen, lossFunctionChosen, known)
    parameters <- updateParameters(parameters, grads, step)
    losses     <- c(losses,loss(fPass$A[[5]], known))
    print(paste0("epoch: ", e, " row:, ",row))
    
  }
  
  epochLoss <- mean(losses)
  es<-c(es,e)
  epochlosses<-c(epochlosses,epochLoss)

  
}


# Convert epoch losses to a data frame for ggplot
epoch_data <- data.frame(epoch = es, epoch_loss = epochlosses)

# Plotting the epoch loss
ggplot(epoch_data, aes(x = epoch, y = epoch_loss)) +
  geom_line() +  # Use geom_line for a line plot
  labs(title = "Training Loss per Epoch",
       x = "Epoch",
       y = "Average Loss") +
  theme_minimal()
# *************************************************************************************************
# Test Net ****************************************************************************************
# *************************************************************************************************

training.accuracy<-testNet(training)
training.accuracy
testing.accuracy<-testNet(testing)
testing.accuracy



plotClassificationMatrix (testing,  title = "Classification Matrix")




# *************************************************************************************************
# Save NN ------------------------------------------------------------------------------------------
# *************************************************************************************************

# save trained net
list.save(parameters, file = 'NN_pars.RData')
# load in trained net
trained.net <- list.load("NN_pars.RData")

# *************************************************************************************************
# -------------------------------------------------------------------------------------------------
# *************************************************************************************************