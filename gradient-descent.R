## NLL and its gradient

sigm <- function(x) {
  return(1 / (1 + exp(-x)))
}

nll  <- function(w, y, x) {
  res <- 0
  for (i in 1:length(y)) {
    pr  <- (w %*% x[i,])
    mu <- sigm(pr)
    if (y[i] == 1) {
      res <- res + log(mu)
    } else { # y = 0
      res <- res + log(1 - mu)
    }
  }
  res <- -1 * res
  res <- res + reg * w %*% w
  return(res)
}

gradNll <- function(w, y, x, reg) {
  as.numeric(t(x) %*% (sigm(x %*% w) - y)) + reg * 2 * w
}

predError  <- function(w, y, x) {
  errs <- sigm(x %*% w) - y
}

## Returns gradient as a function of only w, fixing the data and the regularization.
produceFixedGrad <- function(y, x, reg) {
  return(
    function(w) {
      return(gradNll(w, y, x, reg))
    }
  )
}


#######
## Standard gradient descent, fixed step size

gradientDescent <- function(w, grad, stepSize, iters, tolerance=0) {
  notDone <- TRUE
  while (notDone) {
    w <- w - stepSize * grad(w)
    iters <- iters-1
    notDone <- iters > 0 && sqrt(w %*% w) > tolerance
  }
  return(w)
}


#######
## Stochastic gradient descent

produceStochasticGradients <- function(y, x, batchSize, reg, gr=gradNll) {
  if (batchSize == 1) { # Slice of one row of matrix is not by default a matrix, so a special case is needed :(
    return(
      function(w) {
        ind <- sample(1:length(y), 1)
        xx <- matrix(nrow=1, x[ind,])
        gr(w, y[ind], xx, reg)
      }
    )
  } else {
    return(
      function(w) {
        ind <- sample(1:length(y), batchSize)
        gr(w, y[ind], x[ind,], reg)
      }
    )
  }
}

## Parameter SG is the function produced by the function produceStochasticGradients.
SGD <- function(w, SG, stepSize, iters) {
  for (ii in 1:iters) {
    w <- w - stepSize * SG(w)
    iters <- iters-1
  }
  return(w)
}


#######
## Adam

## This produces a closure to hold the Adam stepsize variables for SGD.
newAdamState <- function(dim) {
  m  <- numeric(dim)
  v <- numeric(dim)
  eps  <- 10^(-8)
  a <- 0.001
  b1 <- 0.9
  b1powt <- 1
  b2 <- 0.999
  b2powt <- 1
  function(g, iter) {
    m <<- b1 * m + (1 - b1) * g
    v <<- b2 * v + (1 - b2) * g * g
    b1powt <<- b1powt * b1
    b2powt <<- b2powt * b2
    return(a * m / (1 - b1powt) / (sqrt(v / (1 - b2powt)) + eps))
  }
}

SGDwithAdam <- function(w, SG, adamState, iters) {
  for (ii in 1:iters) {
    w <- w - adamState(SG(w), ii)
    iters <- iters-1
  }
  return(w)
}

## Applying the implementations to MNIST data

## Load data
{
  to.read <- file("train-images-idx3-ubyte", "rb")
  readBin(to.read, integer(), n=4, endian="big")
  imgs <- t(matrix(as.integer(readBin(to.read,raw(), size=1, n=28*28*60000, endian="big")),28*28,60000))
  to.read <- file("train-labels-idx1-ubyte", "rb")
  readBin(to.read, integer(), n=2, endian="big")
  labels <- as.integer(readBin(to.read, raw(), n=60000, endian="big"))
  imgs <- imgs[labels==5 | labels==6, ]
  imgs <- cbind(rep(1, nrow(imgs)), imgs) # Intercept / bias
  labels <- labels[labels==5 | labels==6]
  labels <- labels - 5 # 5 -> 0, 6 -> 1 for binary regression
  split <- c(4396, 8792) # Split points for training, validation and testing data
  imgs.train <- imgs[1:split[1],]
  imgs.validate <- imgs[(1 + split[1]):split[2],]
  imgs.test <- imgs[(1+split[2]):length(labels),]
  labels.train <- labels[1:split[1]]
  labels.validate <- labels[(1 + split[1]):split[2]]
  labels.test <- labels[(1+split[2]):length(labels)]
}


#######
## Standard gradient descent, plots with respect to iterations

set.seed(341)
w  <- rep(0, ncol(imgs.train))
grad <- produceFixedGrad(labels.train, imgs.train, reg=0)
trainingLoss <- validationLoss <- c()
totalIters <- 1500
for (tt in 1:totalIters) {
  if (tt %% 10 == 0) {cat("Iter ", tt, "\n")}
  w <- gradientDescent(w, grad, stepSize=0.01, iters=1, tolerance=0)
  trainingLoss <- c(trainingLoss, sum(abs(predError(w, labels.train, imgs.train)))/length(labels.train))
  validationLoss <- c(validationLoss, sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate))
}
png("Ex-1-1-iters-341-init-0.png")
plot(1:totalIters, validationLoss, type="l", ylim=c(0,max(c(trainingLoss, validationLoss))), col="red", ylab="Classification error", xlab="number of iterations", main="Step size=0.01")
lines(c(0, totalIters), c(0,0))
legend("topright", legend = c("training loss", "validation loss"),
       text.width = strwidth("validation loss"),
       lty = c(1,1), xjust = 1, yjust = 1, col=c("green", "red"))
lines(1:totalIters, trainingLoss, type="l", col="green")
dev.off()
unregBest <- w

set.seed(314)
w  <- rep(0, ncol(imgs.train))
grad <- produceFixedGrad(labels.train, imgs.train, reg=0)
trainingLoss <- validationLoss <- c()
totalIters <- 1500
for (tt in 1:totalIters) {
  if (tt %% 10 == 0) {cat("Iter ", tt, "\n")}
  w <- gradientDescent(w, grad, stepSize=0.00011, iters=1, tolerance=0)
  trainingLoss <- c(trainingLoss, sum(abs(predError(w, labels.train, imgs.train)))/length(labels.train))
  validationLoss <- c(validationLoss, sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate))
}
png("Ex-1-1-iters-314-init-0-stepS-s.png")
plot(1:totalIters, validationLoss, type="l", ylim=c(0,max(c(trainingLoss, validationLoss))), col="red", ylab="Classification error", xlab="number of iterations", main="Step size=0.00011")
lines(c(-1, totalIters), c(0,0))
legend("topright", legend = c("training loss", "validation loss"),
       text.width = strwidth("validation loss"),
       lty = c(1,1), xjust = 1, yjust = 1, col=c("green", "red"))
lines(1:totalIters, trainingLoss, type="l", col="green")
dev.off()


#######
## Standard gradient descent, plots with respect to step sizes

set.seed(314)
exps <- -6:3
stepSizes <- rev(10^exps)
grad <- produceFixedGrad(labels.train, imgs.train, reg=0)
trainingLoss <- validationLoss <- c()
totalIters <- 1500
for (step in stepSizes) {
  w  <- rep(0, ncol(imgs.train))
  w <- gradientDescent(w, grad, stepSize=step, iters=totalIters, tolerance=0)
  trainingLoss <- c(trainingLoss, sum(abs(predError(w, labels.train, imgs.train)))/length(labels.train))
  validationLoss <- c(validationLoss, sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate))
  ready <- length(trainingLoss)
  readyInd <- 1:ready
  png(paste0("Ex-1-2-step-sizes-", ready, ".png"))
  plot(exps[readyInd], validationLoss, type="l", ylim=c(0,max(c(trainingLoss, validationLoss))), col="red", ylab="Classification error", xlab="log10(step size)", main="Final loss with different step sizes")
  lines(exps[readyInd], rep(0,ready))
  lines(exps[readyInd], trainingLoss, type="l", col="green")
  legend("left", legend = c("training loss", "validation loss"),
         text.width = strwidth("validation loss"),
         lty = c(1,1), xjust = 1, yjust = 1, col=c("green", "red"))
  dev.off()
}

set.seed(314)
exps <- c(exps, 4:6)
stepSizes <- 10^(4:6)
grad <- produceFixedGrad(labels.train, imgs.train, reg=0)
totalIters <- 1500
for (step in stepSizes) {
  w  <- rep(0, ncol(imgs.train))
  w <- gradientDescent(w, grad, stepSize=step, iters=totalIters, tolerance=0)
  trainingLoss <- c(trainingLoss, sum(abs(predError(w, labels.train, imgs.train)))/length(labels.train))
  validationLoss <- c(validationLoss, sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate))
  ready <- length(trainingLoss)
  readyInd <- 1:ready
  png(paste0("Ex-1-2-step-sizes-", ready, ".png"))
  plot(exps[readyInd], validationLoss, type="b", ylim=c(0,max(c(trainingLoss, validationLoss))), col="red", ylab="Classification error", xlab="log10(step size)", main="Final loss with different step sizes")
  lines(exps[readyInd], rep(0,ready))
  lines(exps[readyInd], trainingLoss, type="b", col="green")
  legend("left", legend = c("training loss", "validation loss"),
         text.width = strwidth("validation loss"),
         lty = c(1,1), xjust = 1, yjust = 1, col=c("green", "red"))
  dev.off()
}


#######
## SGD with Adam

set.seed(341)
adam <- newAdamState(785)
SGrad <- produceStochasticGradients(labels.train, imgs.train, batchSize=10, reg=0, gr=gradNll)
w  <- rep(0, ncol(imgs.train))
trainingLoss <- validationLoss <- c()
totalIters <- 1500
nmbrPoints <- 100
pts <- floor(seq(1,totalIters,length.out=nmbrPoints))
for (tt in pts) {
  cat("Iter ", tt, "\n")
  w <- SGDwithAdam(w, SGrad, adam, iter=tt)
  trainingLoss <- c(trainingLoss, sum(abs(predError(w, labels.train, imgs.train)))/length(labels.train))
  validationLoss <- c(validationLoss, sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate))
}

png("Ex-1-3-Adam-341.png")
plot(pts, validationLoss, type="l", ylim=c(0,max(c(trainingLoss, validationLoss))), col="red", ylab="Classification error", xlab="number of iterations", main="SGD with Adam, minibatches of 10")
lines(pts, rep(0, nmbrPoints))
legend("topright", legend = c("training loss", "validation loss"),
       text.width = strwidth("validation loss"),
       lty = c(1,1), xjust = 1, yjust = 1, col=c("green", "red"))
lines(pts, trainingLoss, type="l", col="green")
dev.off()

#######
## Regularization
set.seed(134)
regs <- c(seq(0,2, length.out = 10), seq(3, 25, length.out=20), seq(26, 50, length.out=5)
          )#, 75, 100, 150, 200)
repeats <- 1 # Averaging results over this many tries
## 2 data points
trainingLoss2 <- validationLoss2 <- rep(0, length(regs))
for (tt in 1:repeats) {
  tiny <- c(sample(size=1, which(labels.train==0)), sample(size=1, which(labels.train==1)))
  for (i in 1:length(regs)) {
    rr <- regs[i]
    w  <- rep(0, ncol(imgs.train))
    w <- gradientDescent(w = w, grad = produceFixedGrad(y = labels.train[tiny], x = imgs.train[tiny,], reg = rr), stepSize = 0.001, iters = 500, tolerance = 0.001)
    trainingLoss2[i] <- trainingLoss2[i] + sum(abs(predError(w, labels.train[tiny], imgs.train[tiny,])))/length(labels.train[tiny])
    validationLoss2[i] <-validationLoss2[i] + sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate)
  }
  trainingLoss2 <- trainingLoss2 / repeats
  validationLoss2 <- validationLoss2 / repeats
}
save(file = "losses2.Rdat", list = c("validationLoss2", "trainingLoss2"))
## 10 data points
trainingLoss10 <- validationLoss10 <- rep(0, length(regs))
for (tt in 1:repeats) {
  tiny <- c(sample(size=5, which(labels.train==0)), sample(size=5, which(labels.train==1)))
  for (i in 1:length(regs)) {
    rr <- regs[i]
    w  <- rep(0, ncol(imgs.train))
    w <- gradientDescent(w = w, grad = produceFixedGrad(y = labels.train[tiny], x = imgs.train[tiny,], reg = rr), stepSize = 0.001, iters = 750, tolerance = 0.001)
    trainingLoss10[i] <- trainingLoss10[i] + sum(abs(predError(w, labels.train[tiny], imgs.train[tiny,])))/length(labels.train[tiny])
    validationLoss10[i] <-validationLoss10[i] + sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate)
  }
  trainingLoss10 <- trainingLoss10 / repeats
  validationLoss10 <- validationLoss10 / repeats
}
save(file = "losses10", list = c("validationLoss10", "trainingLoss10"))
## 20 data points
trainingLoss20 <- validationLoss20 <- rep(0, length(regs))
for (tt in 1:repeats) {
  tiny <- c(sample(size=10, which(labels.train==0)), sample(size=10, which(labels.train==1)))
  for (i in 1:length(regs)) {
    rr <- regs[i]
    w  <- rep(0, ncol(imgs.train))
    w <- gradientDescent(w = w, grad = produceFixedGrad(y = labels.train[tiny], x = imgs.train[tiny,], reg = rr), stepSize = 0.001, iters = 500, tolerance = 0.001)
    trainingLoss20[i] <- trainingLoss20[i] + sum(abs(predError(w, labels.train[tiny], imgs.train[tiny,])))/length(labels.train[tiny])
    validationLoss20[i] <-validationLoss20[i] + sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate)
  }
  trainingLoss20 <- trainingLoss20 / repeats
  validationLoss20 <- validationLoss20 / repeats
}
save(file = "losses20", list = c("validationLoss20", "trainingLoss20"))

png("Ex-1-4-regularization-134.png")
plot(regs, validationLoss2, type="l", col="red", ylab="Classification error", xlab="l2-regularization parameter", main="Full batch gradient descent, variable regularization", ylim=c(0,+.12+max(c(validationLoss2, validationLoss10, validationLoss20))))
lines(regs, rep(0, length(regs)))
lines(regs, trainingLoss2, type="l", col="green")
lines(regs, validationLoss10, type="l", col="red", lty=2)
lines(regs, trainingLoss10, type="l", col="green", lty=2)
lines(regs, validationLoss20, type="l", col="red", lty=3)
lines(regs, trainingLoss20, type="l", col="green", lty=3)
legend("topright", legend = c("2 samples, training loss", "2 samples, validation loss", "10 samples, training loss", "10 samples, training loss", "20 samples,  training loss", "20 samples, validation loss"),
       text.width = strwidth("20 samples, validation loss"),
       lty = c(1,1,2,2,3,3), xjust = 1, yjust = 1, col=c("green", "red"))
dev.off()


## Averaging the errors over several tries

set.seed(134)
regs <- c(seq(0,2, length.out = 30), seq(2.2, 7.7, length.out = 30), seq(8, 25, length.out=20)
          )#, seq(26, 50, length.out=5)
          #, 75, 100, 150, 200)
repeats <- 100 # Averaging results over this many tries
## 2 data points
trainingLoss2 <- validationLoss2 <- rep(0, length(regs))
for (tt in 1:repeats) {
  tiny <- c(sample(size=1, which(labels.train==0)), sample(size=1, which(labels.train==1)))
  for (i in 1:length(regs)) {
    rr <- regs[i]
    w  <- rep(0, ncol(imgs.train))
    w <- gradientDescent(w = w, grad = produceFixedGrad(y = labels.train[tiny], x = imgs.train[tiny,], reg = rr), stepSize = 0.001, iters = 500, tolerance = 0.001)
    trainingLoss2[i] <- trainingLoss2[i] + sum(abs(predError(w, labels.train[tiny], imgs.train[tiny,])))/length(labels.train[tiny])
    validationLoss2[i] <-validationLoss2[i] + sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate)
  }
}
trainingLoss2 <- trainingLoss2 / repeats
validationLoss2 <- validationLoss2 / repeats
save(file = "lossesAvg2.Rdat", list = c("validationLoss2", "trainingLoss2"))
## 10 data points
trainingLoss10 <- validationLoss10 <- rep(0, length(regs))
for (tt in 1:repeats) {
  tiny <- c(sample(size=5, which(labels.train==0)), sample(size=5, which(labels.train==1)))
  for (i in 1:length(regs)) {
    rr <- regs[i]
    w  <- rep(0, ncol(imgs.train))
    w <- gradientDescent(w = w, grad = produceFixedGrad(y = labels.train[tiny], x = imgs.train[tiny,], reg = rr), stepSize = 0.001, iters = 1000, tolerance = 0)
    trainingLoss10[i] <- trainingLoss10[i] + sum(abs(predError(w, labels.train[tiny], imgs.train[tiny,])))/length(labels.train[tiny])
    validationLoss10[i] <-validationLoss10[i] + sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate)
  }
}
trainingLoss10 <- trainingLoss10 / repeats
validationLoss10 <- validationLoss10 / repeats
save(file = "lossesAvg10", list = c("validationLoss10", "trainingLoss10"))
## 20 data points
trainingLoss20 <- validationLoss20 <- rep(0, length(regs))
for (tt in 1:repeats) {
  tiny <- c(sample(size=10, which(labels.train==0)), sample(size=10, which(labels.train==1)))
  for (i in 1:length(regs)) {
    rr <- regs[i]
    w  <- rep(0, ncol(imgs.train))
    w <- gradientDescent(w = w, grad = produceFixedGrad(y = labels.train[tiny], x = imgs.train[tiny,], reg = rr), stepSize = 0.001, iters = 750, tolerance = 0)
    trainingLoss20[i] <- trainingLoss20[i] + sum(abs(predError(w, labels.train[tiny], imgs.train[tiny,])))/length(labels.train[tiny])
    validationLoss20[i] <-validationLoss20[i] + sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate)
  }
}
trainingLoss20 <- trainingLoss20 / repeats
validationLoss20 <- validationLoss20 / repeats
save(file = "lossesAvg20", list = c("validationLoss20", "trainingLoss20"))

png("Ex-1-4-regularization-134-b-100.png")
plot(regs, validationLoss2, type="l", col="red", ylab="Classification error", xlab="l2-regularization parameter", main=paste0("Average errors for ", repeats, " runs"), ylim=c(0,+.12+max(c(validationLoss2, validationLoss10, validationLoss20))))
lines(regs, rep(0, length(regs)))
lines(regs, trainingLoss2, type="l", col="green")
lines(regs, validationLoss10, type="l", col="red", lty=2)
lines(regs, trainingLoss10, type="l", col="green", lty=2)
lines(regs, validationLoss20, type="l", col="red", lty=3)
lines(regs, trainingLoss20, type="l", col="green", lty=3)
legend("topright", legend = c("2 samples, training loss", "2 samples, validation loss", "10 samples, training loss", "10 samples, training loss", "20 samples,  training loss", "20 samples, validation loss"),
       text.width = strwidth("20 samples, validation loss"),
       lty = c(1,1,2,2,3,3), xjust = 1, yjust = 1, col=c("green", "red"))
dev.off()

bestReg <-     mean(
  regs[which(validationLoss2==min(validationLoss2))], 
  regs[which(validationLoss10==min(validationLoss10))],
  regs[which(validationLoss20==min(validationLoss20))])
cat("The average of the best regularization parameters: ", bestReg, "\n")


#######
## Early stopping

wBest <- w <- rep(0, 785)
grad <- produceFixedGrad(labels.train, imgs.train, reg=0)
notDone <- 0
best <- sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate)
manieth <- 0
while (notDone < 20) {
  w <- gradientDescent(w=w, grad=grad, stepSize=10^-6, iters=1)
  new <- sum(abs(predError(w, labels.validate, imgs.validate)))/length(labels.validate)
  if (new > best) {
    notDone <- notDone + 1
  } else {
    notDone <- 0
    best <- new
    wBest <- w
  }
  manieth <- manieth + 1
}
cat("Stopped after ", manieth, " iterations.\nValidation error was ", best, ", the next would have been ", new, ".\n")


#######
## Comparing methods

cat("Unregularized test classification error is ", sum(abs(predError(unregBest, labels.test, imgs.test)))/length(labels.test), ".\n")

wReg <- rep(0, ncol(imgs.train))
wReg <- gradientDescent(w = w, grad = produceFixedGrad(y = labels.train, x = imgs.train, reg = bestReg), stepSize = 0.001, iters = 1500, tolerance = 0)
cat("l2-regularized test classification error is ", sum(abs(predError(wReg, labels.test, imgs.test)))/length(labels.test), ".\n")

cat("Early stopping test classification error is ", sum(abs(predError(wBest, labels.test, imgs.test)))/length(labels.test), ".\n")


#######
## Examples of misclassified digits

## Function for displaying mnist digits
im <- function(img, label=0) { # display a digit from the mnist data
  if (length(img)>28*28) {img <- img[(length(img)-28*28+1):length(img)]}
  m <- matrix(img, 28, 28)
  m <- m[,28:1]
  if (label!=0) {name <- paste((label+1)/2+5)} else {name <- ""}
  image(m, col=gray(255:1/255), main = name, xaxt = "n", yaxt = "n")
}

## Unregularized model
U <- imgs.test %*% unregBest

ind5 <- which(labels.test == 0)
ind6 <- which(labels.test == 1)

## Misclassified fives
png("Ex-1-5-wrong-5-1.png")
im(imgs.test[which(U == rev(sort(U[ind5]))[1]),])
dev.off()
png("Ex-1-5-wrong-5-2.png")
im(imgs.test[which(U == rev(sort(U[ind5]))[2]),])
dev.off()
png("Ex-1-5-wrong-5-3.png")
im(imgs.test[which(U == rev(sort(U[ind5]))[3]),])
dev.off()
png("Ex-1-5-wrong-5-4.png")
im(imgs.test[which(U == rev(sort(U[ind5]))[4]),])
dev.off()

## Misclassified sixes
png("Ex-1-5-wrong-6-1.png")
im(imgs.test[which(U == sort(U[ind6])[1]),])
dev.off()
png("Ex-1-5-wrong-6-2.png")
im(imgs.test[which(U == sort(U[ind6])[2]),])
dev.off()
png("Ex-1-5-wrong-6-3.png")
im(imgs.test[which(U == sort(U[ind6])[3]),])
dev.off()
png("Ex-1-5-wrong-6-4.png")
im(imgs.test[which(U == sort(U[ind6])[4]),])
dev.off()


## The most certain 5 and 6
png("Ex-1-5-best-5-1-a.png")
im(imgs.test[which(U == sort(U)[1]),])
dev.off()
png("Ex-1-5-best-5-2-a.png")
im(imgs.test[which(U == sort(U)[2]),])
dev.off()
png("Ex-1-5-best-6-1-a.png")
im(imgs.test[which(U == rev(sort(U))[1]),])
dev.off()
png("Ex-1-5-best-6-2-a.png")
im(imgs.test[which(U == rev(sort(U))[2]),])
dev.off()

## The least certain digits
png("Ex-1-5-worst-1-a.png")
im(imgs.test[which(abs(U) == sort(abs(U))[1]),])
dev.off()
png("Ex-1-5-worst-2-a.png")
im(imgs.test[which(abs(U) == sort(abs(U))[2]),])
dev.off()
