import pymc3 as pm
import arviz as az

likelihood_ca19_9 = 0.25
likelihood_fna = 0.11

with pm.Model() as model:
    #pre_test prob 1
    p_1 = pm.Beta('p_1',alpha=9,beta=1)
    o_1 = pm.Deterministic('o_1',(p_1/(1-p_1)))
    l_ca19_9 = pm.HalfNormal('l_ca19_9',sigma=likelihood_ca19_9)

    #prob after CA-19-9
    o_2 = pm.Deterministic('o_2',o_1*l_ca19_9)
    p_2 = pm.Deterministic('p_2',o_2/(o_2+1))

    #prob after FNA
    l_fna = pm.HalfNormal('l_fna',sigma=likelihood_fna)
    o_3 = pm.Deterministic('o_3',o_2*l_fna)
    p_3 = pm.Deterministic('p_3',o_3/(o_3+1))

    model_trace = pm.sample(cores=1,chains=1,draws=10000)

az.plot_trace(model_trace,var_names=['p_1','l_ca19_9','p_2'])
az.plot_trace(model_trace,var_names=['p_2','l_fna','p_3'])



