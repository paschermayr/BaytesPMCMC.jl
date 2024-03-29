############################################################################################
"""
$(TYPEDEF)

Default arguments for PMCMC constructor.

# Fields
$(TYPEDFIELDS)
"""
struct ParticleGibbs{P<:ParticleFilter,M<:MCMC} <: PMCMCKernel
    "Particle Filter kernel to estimate latent trajectory."
    pf::P
    "MCMC kernel to sample continuous model parameter."
    mcmc::M
    function ParticleGibbs(pf::P, mcmc::M) where {P<:ParticleFilter,M<:MCMC}
        @argcheck !isa(pf.tune.referencing, Marginal) "Cannot use Marginal Filter in Conditional setting, use Ancestral or Conditional referencing"
        return new{P,M}(pf, mcmc)
    end
end
function ParticleGibbs(
    filter::F, mcmc::M; kwargs...
) where {F<:ParticleFilterConstructor,M<:MCMCConstructor}
    return PMCMC(ParticleGibbs, filter, mcmc; kwargs...)
end

############################################################################################
"""
$(SIGNATURES)
Propose new parameter with mcmc psampler. If update=true, objective function will be updated with input model and data.

# Examples
```julia
```

"""
function propose!(
    _rng::Random.AbstractRNG,
    pmcmc::ParticleGibbs,
    model::ModelWrapper,
    data::D,
    proposaltune::T = BaytesCore.ProposalTune(model.info.reconstruct.default.output(1.0))
#    temperature::F = model.info.reconstruct.default.output(1.0),
#    update::U=BaytesCore.UpdateTrue(),
) where {D, T<:ProposalTune}
    ## Get trajectory via PF - always update data.latent in model
    _, pf_diagnostics = propose!(_rng, pmcmc.pf, model, data, proposaltune)
    ## Propose new θₜ - if accepted, model is updated accordingly
    _, mcmc_diagnostics = propose!(_rng, pmcmc.mcmc, model, data, proposaltune)
    ## Assign base diagnostics - ℓobjective and predictions are taken from particle filter
    diagnostics = BaytesCore.BaseDiagnostics(
        pf_diagnostics.base.ℓobjective, pf_diagnostics.base.temperature, pf_diagnostics.base.prediction,
        mcmc_diagnostics.base.iter
    )
    ## Return pmcmc output
    return model.val, PMCMCDiagnostics(diagnostics, pf_diagnostics, mcmc_diagnostics)
end

############################################################################################
#export
export ParticleGibbs, propose!
