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
    _rng::Random.AbstractRNG, model::ModelWrapper, data::D,
    Nchains::Integer, temperdefault::BaytesCore.TemperDefault{B, F}
) where {D, B<:BaytesCore.UpdateBool, F<:AbstractFloat}
    return PMCMC(
        _rng,
        constructor.kernel,
        constructor.filter(_rng, model, data, Nchains, temperdefault),
        constructor.mcmc(_rng, model, data, Nchains, temperdefault);
        default=constructor.default,
    )
end
function PMCMC(
    kernel::Type{P}, filter::F, mcmc::M; kwargs...
) where {P<:PMCMCKernel,F<:ParticleFilterConstructor,M<:MCMCConstructor}
    return PMCMCConstructor(kernel, filter, mcmc, PMCMCDefault(; kwargs...))
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
    pfdiag = infer(_rng, diagnostics, pmcmc.kernel.pf, model, data)
    mcmcdiag = infer(_rng, diagnostics, pmcmc.kernel.mcmc, model, data)
    return PMCMCDiagnostics{pfdiag,mcmcdiag}
end

"""
$(SIGNATURES)
Infer type of predictions of kernel.

# Examples
```julia
```

"""
function infer(
    _rng::Random.AbstractRNG, pmcmc::PMCMC, model::ModelWrapper, data::D, alldata::Bool=true
) where {D}
    if alldata
        return infer(_rng, pmcmc.kernel.pf, model, data)
    else
        return infer(_rng, pmcmc.kernel.mcmc, model, data)
    end
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

function get_tagged(pmcmc::PMCMC)
    return get_tagged(pmcmc.kernel.mcmc)
end

function get_loglik(pmcmc::PMCMC)
    return get_loglik(pmcmc.kernel.pf)
end

function get_prediction(diagnostics::PMCMCDiagnostics)
    return get_prediction(diagnostics.pf)
end

function get_phase(pmcmc::PMCMC)
    return get_phase(pmcmc.kernel.mcmc)
end

function get_iteration(pmcmc::PMCMC)
    return get_iteration(pmcmc.kernel.mcmc)
end

############################################################################################
#export
export PMCMCConstructor, infer
