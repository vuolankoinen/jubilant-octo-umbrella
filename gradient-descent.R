####################
## Defining some functions first

sigm <- function(x) {
    return(1 / (1+exp(-x)))
}

## Negative log-likelihood with l2-regularization
## w weights, x covariates, y class labels, reg regularization parameter
nll  <- function(w, x, y, reg=0) {
    tulos <- 0
    for (ii in 1:length(y)) {
        if (y[ii]) {
            tulos <- tulos + log(sigm(w %*% x[ii,]))
        } else {
            tulos <- tulos + log(1 - sigm(w %*% x[ii,]))
        }
        tulos  <- w %*% w * reg - tulos
        return(tulos)
    }
}

## Gradient of negative log likelihood
gradNll  <- function(w, x, y, reg=0) {
    tulos <- rep(0, length(w))
    for (ii in 1:length(y)) {
        tulos  <- tulos + (sigm(w %*% x[ii,]) - y[ii]) * x[ii,]
    }
    tulos <- tulos/length(w) + 2 * reg * w
    return (tulos)
}

## Classification error
clError <- function(w, X, y) {
    pred <- X %*% w
    res <- abs(y - round(sigm(pred)))
    return(sum(res) / length(y))
}

## (Standard, full-batch) gradient descent
gradientDescent  <- function(w, grad, x, y,
                             reg=0, maxIters=100000,
                             tolerance=0.01, stepSize=0.01) {
    kesken  <- TRUE
    tulos  <- w
    while (kesken) {
        g <- grad(tulos, x, y, reg)
        tulos  <- tulos - stepSize  * g
        maxIters  <- maxIters - 1
        kesken  <- sum(g * g) > tolerance && maxIters != 0
#        cat("G: ", sum(g*g),"\n")
#        for (s in g) {cat(s, ";")}
#        cat("\n")
    }
    return(tulos)
}

## Stochastic gradient optimization
SGD  <- function(w, grad, x, y,
                 minibatch=1,
                 reg=0,
                 maxIters=100000,
                 tolerance=0.01,
                 step=function(y, x){y/x}
                 ) {
    kesken  <- TRUE
    iter  <- 0
    tulos  <- w
    while (kesken) {
        iter  <- iter + 1
        ind  <- sample(1:length(y), minibatch)
        g <- grad(tulos, x[,ind], y[ind], reg)
        tulos  <- tulos - step(g, iter)
        kesken  <- g %*% g > tolerance && maxIters > iter
    }
}

## Adam:
## This produces a closure to hold the Adam stepsize variables for SGD
newAdamState <- function(dim) {
    m  <- numeric(dim)
    v <- numeric(dim)
    eps  <- 10^(-8)
    a <- 0.001
    b_1 <- 0.9
    b_1_pow_t <- 1
    b_2 <- 0.999
    b_2_pow_t <- 1
    function(g, iter) {
        m <<- b_1 * m + (1 - b_1) * g
        v <<- b_2 * v + (1 - b_2) * g * g
        b_1_pow_t <<- b_1_pow_t * b_1
        b_2_pow_t <<- b_2_pow_t * b_2
        return(a * m / (1 - b_1_pow_t) / (sqrt(v / (1 - b_2_pow_t)) + eps))
    }
}

pred <- function(w, testdata) {
  tul <- sigm(testdata %*% w)
  cbind(round(tul), tul)
}

im <- function(img, label=-1) { # display a digit from the mnist data
  if (length(img)>28*28) {img <- img[(length(img)-28*28+1):length(img)]}
  m <- matrix(img, 28, 28)
  m <- m[,28:1]
  if (label>-1) {name <- paste(label)} else {name <- ""}
  image(m, col=gray(255:1/255), ylab = name, xaxt = "n", yaxt = "n")
}

########################
## Putting the functions to use

set.seed(134)

{ # Set up data
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
  
  split <- c(4396, 8792)
  imgs.train <- imgs[1:split[1],]
  imgs.validate <- imgs[(1 + split[1]):split[2],]
  imgs.test <- imgs[(1+split[2]):length(labels),]
  labels.train <- labels[1:split[1]]
  labels.validate <- labels[(1 + split[1]):split[2]]
  labels.test <- labels[(1+split[2]):length(labels)]
}

side  <- 28
w  <- rnorm(side^2+1, sd=0.1)

pdf("Standard-gradient-descent.pdf")

{
  iterations <- 100
  par <- w
  empir_loss <- numeric(iterations)
  val_loss <- numeric(iterations)
  for (tt in 1:iterations) {
    par <- gradientDescent(par, gradNll, imgs.train, labels.train, reg=0.0, maxIters=1, tolerance=0.01, stepSize=0.01)
    empir_loss[tt] <- clError(par, imgs.train, labels.train)
    val_loss[tt] <- clError(par, imgs.validate, labels.validate)
  }
}

plot(1:iterations, empir_loss, col="gray", type="l", xlab = "iteration", ylab = "Classification error")
lines(1:iterations, val_loss, col="blue")
legend("topright", legend = c("training", "validation"),
       text.width = strwidth("1,000,000"),
       lty = c(1,1), xjust = 1, yjust = 1, col=c("gray", "blue"))

dev.off()

################################
## for testing

## Yliyksinkertaiset testiä varten
w <- c(0.1, -0.2) # vakio, yksi parametri
imgs.train <- matrix(nrow = 4, ncol = 2)
imgs.train[,1] <- rep(1,nrow(imgs.train))
imgs.train[,2] <- c(0.1, 0.9, 8.2, 15.6)
labels.train <- c(1,1,0,0)
