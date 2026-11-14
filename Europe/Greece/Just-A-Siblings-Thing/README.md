# Just a siblings' thing

We tested Ares, Athena, and Apollo on the same 1,000 encounters using the supplied weights and the required 0.50 threshold. All three models had 95% accuracy, so we compared how many real catastrophic threats each one missed.

Ares missed **5 out of 100** threats, Athena missed **25**, and Apollo missed **45**. We chose Ares because missing a dangerous encounter is more costly than raising a false alarm, giving the flag **`SIGMOID_ARES`**.