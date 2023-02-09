# BaytesPMCMC

<!---
![logo](docs/src/assets/logo.svg)
[![CI](xxx)](xxx)
[![arXiv article](xxx)](xxx)
-->

[![Documentation, Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://paschermayr.github.io/BaytesPMCMC.jl/)
[![Build Status](https://github.com/paschermayr/BaytesPMCMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/paschermayr/BaytesPMCMC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/paschermayr/BaytesPMCMC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/paschermayr/BaytesPMCMC.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

BaytesPMCMC.jl is a library to perform particle MCMC proposal steps for parameter in a `ModelWrapper` struct, see [ModelWrappers.jl](https://github.com/paschermayr/ModelWrappers.jl).

## Introduction

BaytesPMCMC.jl implements a Particle Gibbs as well as an Particle Metropolis sampler. Note that the latter does not need a specified log objective function for the acceptance rate, but instead uses an approximation from a particle filter. It is recommended to use a fixed stepsize for this sampler, or, if possible, use Particle Gibbs instead.

Let us start with creating a univariate normal Mixture model with two states via ModelWrappers.jl:
```julia
using ModelWrappers, BaytesMCMC, BaytesFilters, BaytesPMCMC
using Distributions, Random, UnPack
_rng = Random.MersenneTwister(1)
N = 10^3
# Parameter
μ = [-2., 2.]
σ = [1., 1.]
p = [.05, .95]
# Latent data
latent = rand(_rng, Categorical(p), N)
data = [rand(_rng, Normal(μ[iter], σ[iter])) for iter in latent]

# Create ModelWrapper struct, assuming we do not know latent
latent_init = rand(_rng, Categorical(p), N)
myparameter = (;
    μ = Param([Normal(-2., 5), Normal(2., 5)], μ, ),
    σ = Param([Gamma(2.,2.), Gamma(2.,2.)], σ, ),
    p = Param(Dirichlet(2, 2), p, ),
    latent = Param([Categorical(p) for _ in Base.OneTo(N)], latent_init, ),
)
mymodel = ModelWrapper(myparameter)
myobjective = Objective(mymodel, data)
```

## Particle Metropolis

Particle Metropolis uses a particle filter to estimate the parameter latent, and an MCMC kernel to estimate all other parameter iteratively. This method is likelihood-free and uses an estimate from the particle filter for the acceptance ratio. As such, one does not need to state the log objective function at all, but gradient based mcmc kernels cannot be used either in this case. To assign a Particle Metropolis sampler, we only have to assign the particle filter dynamics as in BaytesFilters.jl:

```julia
# Assign Model dynamics
function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{BaseModel}})
    @unpack model, data = objective
    @unpack μ, σ, p = model.val

    initial_latent = Categorical(p)
    transition_latent(particles, iter) = initial_latent
    transition_data(particles, iter) = Normal(μ[particles[iter]], σ[particles[iter]])

    return Markov(initial_latent, transition_latent, transition_data)
end
dynamics(myobjective)

# Assign an objective for both a particle filter and an mcmc kernel:
myobjective_pf = Objective(mymodel, data, :latent)
myobjective_mcmc = Objective(mymodel, data, (:μ, :σ, :p))

# Assign Particle Metropolis algorithm
mcmcdefault = MCMCDefault(;
	stepsize = ConfigStepsize(; ϵ = 1.0, stepsizeadaption = UpdateFalse()),
)
pmetropolis = ParticleMetropolis(
    #Particle filter
    ParticleFilter(_rng, myobjective_pf),
    #MCMC kernel
    MCMC(_rng, Metropolis, myobjective_mcmc, mcmcdefault)
)

# Proposal steps work exactly as in BaytesFilters.jl and BaytesMCMC.jl
_val, _diagnostics = propose!(_rng, pmetropolis, mymodel, data)
```

## Particle Gibbs

Particle Gibbs uses a conditional particle filter along with an MCMC kernel. In order to use this sampler, one has to define an objective function. However, we can condition the target function on the
latent sequence, which results usually in a much easier and faster form than the (marginal) likelihood,
where latent variables have to be integrated out. Once defined, we can also use more advanced mcmc kernels for estimation.

```julia
function (objective::Objective{<:ModelWrapper{BaseModel}})(θ::NamedTuple)
    @unpack model, data, tagged = objective
    @unpack μ, σ, p, latent = θ
## Prior -> a faster shortcut without initializing the priors again
    lprior = log_prior(tagged.info.transform.constraint, ModelWrappers.subset(θ, tagged.parameter) )
##Likelihood
    dynamicsᵉ = [Normal(μ[iter], σ[iter]) for iter in eachindex(μ)]
    dynamicsˢ = Categorical(p)
    ll = 0.0
#FOR PMCMC ~ target p(θ ∣ latent_1:t, data_1:t)
    for iter in eachindex(data)
        ll += logpdf(dynamicsᵉ[latent[iter]], data[iter])
        ll += logpdf(dynamicsˢ, latent[iter] )
    end
#=
# FOR MCMC ~ target p(θ ∣ data_1:t) by integrating out latent_1:t
    for time in eachindex(data)
        ll += logsumexp(logpdf(dynamicsˢ, iter) + logpdf(dynamicsᵉ[iter], grab(data, time)) for iter in eachindex(dynamicsᵉ))
    end
=#
    return ll + lprior
end
myobjective_mcmc(myobjective_mcmc.model.val)
# Note - It is good to benchmark this function, as it will allocate >98% of the mcmc kernel time
using BenchmarkTools
$myobjective_mcmc($myobjective_mcmc.model.val) #13.600 μs (2 allocations: 176 bytes)
```

As we can analytically compute the marginal likelihood of a univariate mixture, I could also write down (and comment out) the corresponding objective function in the MCMC case. This should help understanding my comments above. Once our objective is defined, we can intialize a `ParticleGibbs` struct and sample with it:

```julia
# Assign an objective for both a particle filter and an mcmc kernel:
myobjective_pf = Objective(mymodel, data, :latent)
myobjective_mcmc = Objective(mymodel, data, (:μ, :σ, :p))

# Assign Particle Gibbs sampler
pfdefault = ParticleFilterDefault(referencing = Conditional())
pgibbs = ParticleGibbs(
    #Conditional Particle filter
    ParticleFilter(_rng, myobjective_pf, pfdefault
    ),
    #MCMC kernel -> can use more advanced kernels
    MCMC(_rng, NUTS, myobjective_mcmc)
)

# Proposal steps work exactly as in BaytesFilters.jl and BaytesMCMC.jl
_val, _diagnostics = propose!(_rng, pgibbs, mymodel, data)
```

## Going Forward

This package is still highly experimental - suggestions and comments are always welcome!

<!---
# Citing Baytes.jl

If you use Baytes.jl for your own research, please consider citing the following publication: ...
-->
