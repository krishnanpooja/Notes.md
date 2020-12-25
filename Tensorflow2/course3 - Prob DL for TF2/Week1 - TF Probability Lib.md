### Week 1 - TensorFlow Probability library

tensorflow.org/probabilty
```
import tensorflow_probability as tfp
```

 **Univariate Distributions**
 
 Distribution of single random variable
 
 ```
 import tensorflow as tf
 import tensorflow_probability as tfp
 
 tfd = tfp.distributions
 normal = tfd.Normal(loc=0.0, scale=1.) #loc=mean and scale=std
 normal.sample() # returns a single sample basically a tensor
 normal.sample(3) # sample of length 3
 normal.prob(0.5)
 normal.log_prob(0.5) #log prob of the value provided as input
 
 
 bern = tfd.Bernoulli(probs = 0.7)
 bern_1 = tfd.Bernoulli(logits = 0.74)
 bern_1.sample(3) # sample of length 3
 bern_1.prob(0.5)
 bern_1.log_prob(0.5)
 
 batched_bern = tfd.Bernoulli(probs=[0.4,0.5]) #batch_shape = [2] contains batch of two bernoulli distributions
 batched_bern.sample(3) #returns Tensor of shape (3,2)
 ```
 event_shape = captures the dimensionality of the random variable itself. Empty for univariate distribution
 
**Multivariate Distributions**
 
 Multiple Random Variable
  
```
import tensorflow as tf
import tensorflow_probability as tfp
 
tfd = tfp.distributions
normal = tfd.MultivariateNormalDiag(loc=[-1.,0.5], scale_diag=[1.,1.5]) #loc=mean and scale=std event_shape -[2] -> 2D 
normal.event_shape
 
normal.sample(3) # [3,2] tensor
normal.log_prob([-0.2,1.8]) #tf.Tensor(-2.938, hhape=(), dtype=float32)

#batched multivariate distribution
normal = tfd.MultivariateNormalDiag(loc=[[-1.,0.5], [2,0],[-0.5,1.5]], scale_diag=[[1.,1.5], [2,0.5],[1,1]]) 
normal.sample(3) # (2,3,2) (sample_size,batch_size,event_shape)


```
 The batch_shape = [2] -> Univariate normal distribution
 event_shape = [2] -> Multivariate normal distribution
 
 Difference in evident when we check the log_prob. The Multivariate returns single tensor value.
 
 **Independent Distribution**
 
 Changes univariate into multi variate
 
 ```
import tensorflow as tf
import tensorflow_probability as tfp
 
tfd = tfp.distributions
normal = tfd.Normal(loc=[-1.,0.5], scale=[1,1.5])
independent_normal = tfd.Independent(normal, reinterpreted_batch_ndims=1) 
# reinterpreted_batch_ndims says how many of the batch dim should be absorbed into the event space
```

**Naive Bayes Classifier**
Each feature is independent given the class.





 
  
