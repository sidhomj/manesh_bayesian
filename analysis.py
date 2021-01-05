import pymc3 as pm
import arviz as az
import numpy as np

likelihood_ca19_9 = 0.25
likelihood_fna = 0.11
certainty = 1
alpha = 9*certainty
beta = 1*certainty
mean_prior = alpha/(alpha+beta)
var_prior = (alpha*beta)/(np.power(alpha+beta,2)*(alpha+beta+1))

with pm.Model() as model:
    #pre_test prob 1
    "assuming beta prior with a mean of 0.90"
    p_1 = pm.Beta('p_1',alpha=alpha,beta=beta)

    "assuming uninformed prior belief"
    #p_1 = pm.Uniform('p_1')

    o_1 = pm.Deterministic('o_1',(p_1/(1-p_1)))
    sens_ca19_9 = pm.Uniform('sens_ca19_9',lower=0.70,upper=0.92)
    l_ca19_9 = pm.Deterministic('l_ca19_9',(1-sens_ca19_9)/sens_ca19_9)

    #prob after CA-19-9
    o_2 = pm.Deterministic('o_2',o_1*l_ca19_9)
    p_2 = pm.Deterministic('p_2',o_2/(o_2+1))

    #prob after FNA
    l_fna = likelihood_fna
    o_3 = pm.Deterministic('o_3',o_2*l_fna)
    p_3 = pm.Deterministic('p_3',o_3/(o_3+1))

    model_trace = pm.sample(cores=1,chains=1,draws=10000)

az.plot_trace(model_trace,var_names=['p_1','l_ca19_9','p_2'])
az.plot_trace(model_trace,var_names=['p_2','l_fna','p_3'])
az.plot_trace(model_trace,var_names=['p_1','p_2','p_3'])

np.mean(model_trace['p_3'])
np.percentile(model_trace['p_3'],90)

