data {
  int<lower=0> N;   //Number of Observations
  int<lower=0> y[N];   //Poisson data
}

parameters {
  real<lower=0> lambda; //Poisson Mean
}

model {
  target += gamma_lpdf(lambda | 2,0.25); //Gamma Prior
  //target += uniform_lpdf(lambda | 1,2); //Uniform Prior with Poor Domain
 // target += uniform_lpdf(lambda | 0,100); //Uniform Prior with Better Domain
  
  target += poisson_lpmf(y | lambda);
}

