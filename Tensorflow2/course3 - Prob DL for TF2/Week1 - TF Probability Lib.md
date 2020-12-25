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
 normal = tfd.Normal(loc=0.0, scale=1.)
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
 
  **Univariate Distributions**
