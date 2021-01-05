import numpy as np

prior_p = 0.90
prior_odds = prior_p/(1-prior_p)
lr_ca19_9 = 0.25

post_ca19_9_odds = prior_odds*lr_ca19_9
post_ca19_9_p = post_ca19_9_odds/(post_ca19_9_odds+1)
print(post_ca19_9_p)

lr_fna = 0.11
post_fna_odds = post_ca19_9_odds*lr_fna
post_fna_p = post_fna_odds/(post_fna_odds+1)
print(post_fna_p)