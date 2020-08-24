##########
#Movielens Capstone Project
##########
#
#Code in this R script mirrors the Rmd(R-Markdown) document to include only the R scripts 
#that is in the document.
#
##########
#Dataset
##########
#Movielens 10M dataset can be found in the links below:
#
#https://grouplens.org/datasets/movielens/10m/
#http://files.grouplens.org/datasets/movielens/ml-10m.zip
#
##########
#Creating Train and validation sets
##########

if(!require(tidyverse)) 
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) 
  install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) 
  install.packages("data.table", repos = "http://cran.us.r-project.org")

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", 
                             readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")
# 'Validation' set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, 
                                  times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
# Make sure userId and movieId in 'validation' set are also in 'edx' set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
# Add rows removed from 'validation' set back into 'edx' set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########
#Creating train sets and tests sets for model building
##########

#p set to 10% for 10% of edx's data
set.seed(1, sample.kind = "Rounding")
test_ind<- createDataPartition(y=edx$rating,p=0.1,times=1, list= FALSE)
train_set<- edx[-test_ind,]
pre_test<-edx[test_ind,]

#Include in movieId and userId for data analysis and model building later

test_set<- pre_test %>% semi_join(train_set, by = "movieId")%>%
  semi_join(train_set, by = "userId")
removed<-anti_join(pre_test,test_set)
train_set<-rbind(train_set,removed)

rm(test_ind,pre_test,removed)

##########
#General statistics/analysis
##########

#first 6 lines of data
head(edx)

#view column names of dataset as well as a preview of the content
str(edx)

#view dimensions of dataset
dim(edx)

#####
#Date
#####

#First, convert and determine start and end date of data collection period with the help of lubridate
if(!require(lubridate)) 
  install.packages("lubridate", repos = "http://cran.us.r-project.org")
tibble(`Start Date` = date(as_datetime(min(edx$timestamp), origin="1970/01/01")),
       `End Date` = date(as_datetime(max(edx$timestamp), origin="1970/01/01"))) %>%
  mutate(Collection_Period = duration(max(edx$timestamp)-min(edx$timestamp)))

#Next, create a simple distribution to show how many ratings are within different years
edx_timestamp_dist<- edx %>% mutate(year = year(as_datetime(timestamp,origin="1970/01/01"))) %>%
  ggplot(aes(x=year)) + scale_y_continuous() + 
  geom_histogram(binwidth = 0.50, color = "white") + 
  ggtitle("Movielens Ratings Distribution by Year") + 
  ylab("# of Ratings") + xlab("Year")
edx_timestamp_dist

#####
#Movie
#####

#Find out number of distinct movies in the dataset (10677)
n_distinct(edx$movieId)

#Create a distribution to show the number of ratings of movies.
edx_movie_dist <- edx %>% group_by(movieId) %>%
  summarize(n=n()) %>%
  ggplot(aes(n)) + scale_x_log10() +
  geom_histogram(binwidth = .20, color = "white") +
  ggtitle("Movielens Distribution of Movies and Ratings") + 
  ylab("# of Movies") + xlab("# of Ratings")
edx_movie_dist

#####
#Users
#####

#Find out number of distinct users in the dataset (69878).
n_distinct(edx$userId)

#Create a distribution of the number of ratings per user(rater)
edx_userid_dist <- edx %>% group_by(userId) %>%
  summarise(n = n()) %>%
  ggplot(aes(n)) + scale_x_log10() +
  geom_histogram(binwidth = 0.10, color = "white") +
  ggtitle("Movielens Distribution of Users and Ratings") +
  ylab("# of Users") + xlab("# of Ratings")
edx_userid_dist

#####
#Ratings
#####

#Find out number of unique ratings in database (10)
n_distinct(edx$rating)

#Group by rating and summarize number of unique ratings per each rating level
edx %>% group_by(rating) %>% summarize(n = n())

#Create line chart showing distribution of the count of ratings per each rating level (0.5 to 5.0).
edx_rating_dist <- edx %>% group_by(rating) %>% summarize(n = n()) %>%
  ggplot(aes(x = rating, y = n)) + 
  geom_point() + geom_line() +
  scale_y_log10() + 
  ggtitle("Movielens Rating Distribution") + 
  ylab("Count") + xlab("Rating")

edx_rating_dist

#####
#Genres
#####

#Find out number of distinct genres (797)
n_distinct(edx$genres)

#We will not be using genre for our data exploration and model building.

#Selecting columns that are of interest to our project
train_set<- train_set %>% select(userId, movieId, title, rating)
test_set<- test_set %>% select(userId, movieId, title, rating)

##########
#Model preparation
##########
#
#Model loss function
#

#We first code in the loss functions 
#MSE (Mean squared error)
MSE <- function(true_ratings, predicted_ratings){
  mean((true_ratings - predicted_ratings)^2)
}

#RMSE (Root mean squared error)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#####
#Random prediction
#####

#Random prediction can be done with monte carlo simulation


set.seed(1234, sample.kind = "Rounding")

#Generate probability for each rating.
prob <- function(x , y) mean(y == x)
ratings <- seq(0.5, 5, 0.5)

#Monte Carlo Simulation
B <- 1000
MC <-replicate(B, {
  sim <- sample(100, replace = TRUE, train_set$rating)
  sapply(ratings, prob, y = sim)
})

probs<- sapply( 1:nrow(MC), function(x) mean(MC[x,]) )

#Random Prediction
y_random <- sample(ratings, size = nrow(test_set), replace=TRUE, prob = probs)

#Create tibble table to present errors
results<- tibble()

#Combine the first random prediction into the table
results<- bind_rows(results, 
                    tibble(Methods ="Random Prediction", 
                           RMSE = RMSE(test_set$rating,y_random),
                           MSE = MSE(test_set$rating,y_random)))

results %>% knitr::kable()

#####
#Linear Model
#####

#We start of with the simplest formula and build onto it with user effect, movie effect,
#then later on regularization and matrix factorization.

#We start off with using only the mean of values
mu <- mean(edx$rating)
mu

#Combine the mean method into the table
results <- bind_rows(results, 
                     tibble(Methods = "Mean Only Model",
                            RMSE = RMSE(test_set$rating,mu),
                            MSE = MSE(test_set$rating, mu)))

#View results table (we will build onto this table)
results %>% knitr::kable()

#####
#Adding in movie effect
#####

#Group and create b_i(movie effect)
movie_avgs <- train_set %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))

#Plot histogram to show distribution
movie_avgs %>% ggplot(aes(x = b_i)) +
  geom_histogram(binwidth = .25, color =I("black")) + 
  ggtitle("Movie Effect's Distribution") + 
  ylab("Count") + xlab("Movie Effect") +
  scale_y_continuous()

#Predict the rating with b_i and the mean
yh_b_i <- mu + test_set %>% left_join(movie_avgs, by="movieId") %>% .$b_i

#Calculate the RMSE and input into table
results <- bind_rows(results,
                     tibble(Methods = "Movie Effect Model",
                            RMSE = RMSE(test_set$rating,yh_b_i),
                            MSE = MSE(test_set$rating,yh_b_i)))

results %>% knitr::kable()

#####
#Adding in user effect
#####

#Group by userId, join in b_i, and create b_u(User Effect)
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#Plot histogram to show distribution
train_set %>% group_by(userId) %>% summarize(b_u = mean(rating)) %>%
  ggplot(aes(x = b_u)) + 
  geom_histogram(binwidth=.25, color = I("black")) +
  ggtitle("User Effect's Distribution") +
  ylab("Count") + xlab("User Effect") + scale_y_continuous()

#Predict ratings with b_i and b_u and mu
yh_b_i_u <- test_set %>% left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>% 
  mutate(predictions = mu + b_i + b_u) %>% .$predictions

#Calculate RMSE and put into table
results<- bind_rows(results,
                    tibble(Methods = "Mean with Movie and User Effect Model",
                           RMSE = RMSE(test_set$rating, yh_b_i_u),
                           MSE = MSE(test_set$rating, yh_b_i_u)))

results %>% knitr::kable()

#####
#Regularization
#####


#Create function for regularization

regularization_func <- function(lambda, train, test){
  
  #mean stays the same as it does not change
  mu<- mean(train$rating)
  
  #Movie Effect
  b_i <- train %>%  group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n() + lambda))
  
  #User Effect
  b_u <- train %>% left_join(b_i, by = "movieId") %>%
    group_by(userId) %>% 
    summarize(b_u = sum( rating - b_i - mu )/(n() + lambda)) 
  
  #Prediction with mu, b_i, and b_u
  pred_rate <- test %>% left_join(b_i, by="movieId") %>%
    left_join(b_u,by="userId") %>%
    mutate(preds = mu + b_i + b_u) %>%
    pull(preds)
  
  return(RMSE(pred_rate, test$rating))
}
#define lambda (tuning parameter)
lambdas <- seq( 0, 10, 0.25 )

#Tune lambda (made take couple minutes)
rmses <- sapply(lambdas, regularization_func, train= train_set, test = test_set)

#Plot lambda vs RMSE to find lowest RMSE penalization
tibble(Lambda = lambdas, RMSE = rmses) %>% ggplot(aes(Lambda, RMSE)) +
  geom_point()+
  ggtitle("Regularization Plot")

#Pick the lowest RMSE lambda from the plot above
lambda <- lambdas[which.min(rmses)]

#Prediction is then calculated using regularized parameters
mu<- mean(train_set$rating)

#b_i
movie_avgs <- train_set %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n() + lambda))

#b_u
user_avgs <- train_set %>% left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n() + lambda))

#Prediction and add to table
yh_reg_u_i <- test_set %>% left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred_rate = mu + b_i + b_u) %>%
  pull(pred_rate)

#Update table with calculated RMSE/MSE
results <- bind_rows(results, 
                     tibble(Methods = "Regularized User and Movie Effect Model",
                            RMSE = RMSE(test_set$rating,yh_reg_u_i),
                            MSE = MSE(test_set$rating,yh_reg_u_i)))

results %>% knitr::kable()

#####
#Matrix Factorization
#####
#
#We use a package here called recosystem. It is especially helpful to break down data into matrices
#and estimate ratings using parallel matrix factorization.
#
#Link: https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html
#
#Below instructions carefully follow instructions laid out in the website above

#Download recosystem
if(!require(recosystem)) 
  install.packages("recosystem", repos = "http://cran.us.r-project.org")

#Random algorithm
set.seed(1234, sample.kind = "Rounding")

#1. Convert train and test sets to recosystem format
train_dat <- with(train_set, data_memory(user_ind = userId, item_ind = movieId, rating = rating))
test_dat <- with(test_set, data_memory(user_ind = userId, item_ind = movieId, rating = rating))

#2. Create object
r<- Reco()

#3. Select tuning parameter (Takes up to a hour)
opts <- r$tune(train_dat, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                      costp_l1 = c(0.01, 0.1), 
                                      costq_l1 = c(0.01, 0.1),
                                      nthread = 4, niter = 10))
opts

#Train algorithm
r$train(train_dat, opts = c(opts$min, nthread = 4, niter = 20))

#Prediction and add to table (takes a couple minutes)
yh_r<- r$predict(test_dat, out_memory())

head(yh_r,10)

#Calculate RMSE/MSE and update to table
results <- bind_rows(results, 
                     tibble(Methods = "Matrix Factorization via Recosystem Method",
                            RMSE = RMSE(test_set$rating, yh_r),
                            MSE = MSE(test_set$rating,yh_r)))

results %>% knitr::kable()

############
#Validation
############
#From the results generated from matrix factorization and regularization resulted in the lowest RMSE.
#Therefore, we will use the two to calculate final RMSE by training edx and testing validation set.

#
#Linear model with regularization
#

#mean only
edx_mu <- mean(edx$rating)

# b_i
edx_mov_avgs <- edx %>% group_by(movieId) %>%
  summarize(b_i = sum(rating - edx_mu)/(n() + lambda))

# b_u
edx_use_avgs <- edx %>% left_join(edx_mov_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - edx_mu)/(n() + lambda))

# Predictions and add to table
edx_yh_rate <- validation %>% left_join(edx_mov_avgs, by = "movieId") %>%
  left_join(edx_use_avgs, by = "userId") %>%
  mutate(pred_rate = edx_mu + b_i + b_u) %>%
  pull(pred_rate)

results <- bind_rows(results, 
                     tibble(Methods = "Final Regularization Model", 
                            RMSE = RMSE(validation$rating, edx_yh_rate),
                            MSE  = MSE(validation$rating, edx_yh_rate)))

results %>% knitr::kable()

#Show top 10 movies as demonstration of model


# Top 5 best movies:
validation %>%
  left_join(edx_mov_avgs, by = "movieId") %>%
  left_join(edx_use_avgs, by = "userId") %>%
  mutate(pred_rate = edx_mu + b_i + b_u) %>%
  arrange(-pred_rate) %>%
  group_by(title) %>%
  select(title) %>%
  head(5)

# Top 5 worst movies:
validation %>%
  left_join(edx_mov_avgs, by = "movieId") %>%
  left_join(edx_use_avgs, by = "userId") %>%
  mutate(pred_rate = edx_mu + b_i + b_u) %>%
  arrange(pred_rate) %>%
  group_by(title) %>%
  select(title) %>%
  head(5)
#
#Model with matrix factorization
#

#Random algorithm
set.seed(1234, sample.kind="Rounding")

#Convert edx data to recosystem format
edx_r <- with(edx, data_memory(user_ind = userId,
                               item_ind = movieId,
                               rating = rating))

#Convert validation set to recosystem format
validation_r <- with(validation, data_memory(user_ind = userId,
                                             item_ind = movieId,
                                             rating = rating))

#Create object
r<- Reco()

#Select tuning parameters (May take up to a hour)
opts <- r$tune(edx_r, opts = list(dim = c(10,20,30),
                                  lrate = c(0.1,0.2),
                                  costp_l2 = c(0.01, 0.1), 
                                  costq_l2 = c(0.01, 0.1),
                                  nthread= 4, niter=10))

#Training algorithm
r$train(edx_r,opts = c(opts$min, nthread=4, niter=20))

#Generating prediction (takes a couple minutes)
yh_f_r<- r$predict(validation_r, out_memory())


#Calculate rmse and update to table
results <- bind_rows(results, 
                     tibble(Methods = "Final Matrix Factorization via Recosystem",
                            RMSE = RMSE(validation$rating, yh_f_r),
                            MSE = MSE(validation$rating,yh_f_r)))

results %>% knitr::kable()

#Show top/worse 5 movies as demonstration of model

# Top 5 best movies:
tibble(title = validation$title, rating = yh_f_r) %>%
  arrange(-rating) %>%
  group_by(title) %>%
  select(title) %>%
  head(5)

# Top 5 worst movies:
tibble(title = validation$title, rating = yh_f_r) %>%
  arrange(rating) %>%
  group_by(title) %>%
  select(title) %>%
  head(5)
