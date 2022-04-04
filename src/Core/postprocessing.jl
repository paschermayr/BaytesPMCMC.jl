############################################################################################
"""
$(SIGNATURES)
Callable struct to make initializing PMCMC sampler easier in sampling library.

# Examples
```julia
```

"""
struct PMCMCConstructor{P,F<:ParticleFilterConstructor,M<:MCMCConstructor,D<:PMCMCDefault} <:
       AbstractConstructor
    "Valid PCMC Kernel."
    kernel::P
    "Particle Filter Constructor."
    filter::F
    "MCMC Constructor"
    mcmc::M
    "All other relevant keywords."
    default::D
    function PMCMCConstructor(
        kernel::Type{P}, filter::F, mcmc::M, default::D
    ) where {P<:PMCMCKernel,F<:ParticleFilterConstructor,M<:MCMCConstructor,D<:PMCMCDefault}
        return new{typeof(kernel),F,M,D}(kernel, filter, mcmc, default)
    end
end
function (constructor::PMCMCConstructor)(
    _rng::Random.AbstractRNG,
    model::ModelWrapper,
    data::D,
    temperature::F,
    info::BaytesCore.SampleDefault
) where {D, F<:AbstractFloat}
    return PMCMC(
        _rng,
        constructor.kernel,
        constructor.filter(_rng, model, data, temperature, info),
        constructor.mcmc(_rng, model, data, temperature, info),
        constructor.default,
        info
    )
end
function PMCMC(
    kernel::Type{P}, filter::F, mcmc::M; kwargs...
) where {P<:PMCMCKernel,F<:ParticleFilterConstructor,M<:MCMCConstructor}
    return PMCMCConstructor(kernel, filter, mcmc, PMCMCDefault(; kwargs...))
end

function get_sym(constructor::PMCMCConstructor)
    #return get_sym(constructor.mcmc)
    ## Obtain symbols from mcmc and particle filter
    sym1 = get_sym(constructor.mcmc)
    sym2 = get_sym(constructor.filter)
    ## Return unique symbols as tuple
    sym = unique((sym1..., sym2...))
    return Tuple(sym)
end

############################################################################################
"""
$(SIGNATURES)
Infer PMCMC diagnostics type.

# Examples
```julia
```

"""
function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    pmcmc::PMCMC,
    model::ModelWrapper,
    data::D,
) where {D}
    TPrediction = infer(_rng, pmcmc.kernel.pf, model, data)
    pfdiag = infer(_rng, diagnostics, pmcmc.kernel.pf, model, data)
    mcmcdiag = infer(_rng, diagnostics, pmcmc.kernel.mcmc, model, data)
    return PMCMCDiagnostics{TPrediction, pfdiag, mcmcdiag}
end

"""
$(SIGNATURES)
Infer type of predictions of kernel.

# Examples
```julia
```

"""
function infer(
    _rng::Random.AbstractRNG, pmcmc::PMCMC, model::ModelWrapper, data::D
) where {D}
    return infer(_rng, pmcmc.kernel.pf, model, data)
end

############################################################################################
function results(
    diagnosticsᵛ::AbstractVector{M}, pmcmc::PMCMC, Ndigits::Integer, quantiles::Vector{T}
) where {T<:Real,M<:PMCMCDiagnostics}
    ## Print MCMC diagnostics
    results(
        [diagnosticsᵛ[iter].mcmc for iter in eachindex(diagnosticsᵛ)],
        pmcmc.kernel.mcmc,
        Ndigits,
        quantiles,
    )
    ## Print PF diagnostics
    results(
        [diagnosticsᵛ[iter].pf for iter in eachindex(diagnosticsᵛ)],
        pmcmc.kernel.pf,
        Ndigits,
        quantiles,
    )
    return nothing
end

############################################################################################
function result!(pmcmc::PMCMC, result::L) where {L<:ℓObjectiveResult}
    result!(pmcmc.kernel.mcmc, result)
    return nothing
end
function get_result(pmcmc::PMCMC)
    return get_result(pmcmc.kernel.mcmc)
end
function predict(_rng::Random.AbstractRNG, kernel::PMCMCKernel, objective::Objective)
    return predict(_rng, kernel.pf, objective)
end
function predict(_rng::Random.AbstractRNG, pmcmc::PMCMC, objective::Objective)
    return predict(_rng, pmcmc.kernel, objective)
end

############################################################################################
#export
export PMCMCConstructor, infer
