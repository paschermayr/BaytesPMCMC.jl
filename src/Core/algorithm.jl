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
    default::PMCMCDefault=PMCMCDefault(),
    info::BaytesCore.SampleDefault = BaytesCore.SampleDefault()
) where {M<:PMCMCKernel}
    ## Assign PMCMC Kernel
    pmcmc = kernel(pf, mcmc)
    ## Assign tuning container ~ Placeholder for now
    tune = PMCMCTune()
    ## Return struct
    return PMCMC(pmcmc, tune)
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
    temperature::F = model.info.flattendefault.output(1.0),
    update::U=BaytesCore.UpdateTrue(),
) where {D,F<:AbstractFloat, U<:BaytesCore.UpdateBool}
    ## Make PMCMC Proposal step
    val, diagnostics = propose!(_rng, pmcmc.kernel, model, data, temperature, update)
    ## Pack and return output
    return val, diagnostics
end

############################################################################################
#export
export PMCMC, PMCMCDefault, propose!
