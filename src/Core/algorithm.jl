############################################################################################
"""
$(TYPEDEF)

Default arguments for PMCMC constructor.

# Fields
$(TYPEDFIELDS)
"""
struct PMCMCDefault
    function PMCMCDefault()
        return new()
    end
end

############################################################################################
"""
$(TYPEDEF)

Particle MCMC container, container kernel and tuning.

# Fields
$(TYPEDFIELDS)
"""
struct PMCMC{M<:PMCMCKernel,N<:PMCMCTune} <: AbstractAlgorithm
    "PMCMC sampler"
    kernel::M
    "Tuning configuration for kernel."
    tune::N
    function PMCMC(kernel::M, tune::N) where {M<:PMCMCKernel,N<:PMCMCTune}
        return new{M,N}(kernel, tune)
    end
end
function PMCMC(
    _rng::Random.AbstractRNG,
    kernel::Type{M},
    pf::ParticleFilter,
    mcmc::MCMC,
    Nchains::Integer = 1,
    temperdefault::BaytesCore.TemperDefault = BaytesCore.TemperDefault(
        BaytesCore.UpdateFalse(),
        mcmc.tune.tagged.info.flattendefault.output(1.0),
    );
    default::D = PMCMCDefault(),
) where {M<:PMCMCKernel,D<:PMCMCDefault}
    ## Checks before algorithm is initiated
    #ArgCheck.@argcheck Nchains == length(temperdefault.val) "Nchains and number of temperatures differ in size."
    ## Assign PMCMC Kernel
    pmcmc = kernel(pf, mcmc)
    ## Assign tuning container ~ Placeholder for now
    tune = PMCMCTune()
    ## Return struct
    return PMCMC(pmcmc, tune)
end
function PMCMC(
    kernel::Type{M},
    pf::ParticleFilter,
    mcmc::MCMC,
    Nchains::Integer = 1,
    temperdefault::BaytesCore.TemperDefault = BaytesCore.TemperDefault(
        BaytesCore.UpdateFalse(),
        mcmc.tune.tagged.info.flattendefault.output(1.0),
    );
    kwargs...
) where {M<:PMCMCKernel}
    return PMCMC(Random.GLOBAL_RNG, kernel, pf, mcmc, Nchains, temperdefault; kwargs...)
end

############################################################################################
"""
$(SIGNATURES)
Propose new parameter with pmcmc sampler. If update=true, objective function will be updated with input model and data.

# Examples
```julia
```

"""
function propose!(
    _rng::Random.AbstractRNG,
    pmcmc::PMCMC,
    model::ModelWrapper,
    data::D,
    update::U=BaytesCore.UpdateTrue(),
) where {D,U<:BaytesCore.UpdateBool}
    ## Make PMCMC Proposal step
    val, diagnostics = propose!(_rng, pmcmc.kernel, model, data, update)
    ## Pack and return output
    return val, diagnostics
end
function propose!(
    pmcmc::PMCMC, model::ModelWrapper, data::D, update::U=BaytesCore.UpdateTrue()
) where {D,U<:BaytesCore.UpdateBool}
    return propose!(Random.GLOBAL_RNG, pmcmc, model, data, update)
end

############################################################################################
#export
export PMCMC, PMCMCDefault, propose!
