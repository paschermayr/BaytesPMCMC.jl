############################################################################################
"""
$(TYPEDEF)

Default arguments for PMCMC constructor.

# Fields
$(TYPEDFIELDS)
"""
struct ParticleMetropolis{P<:ParticleFilter,M<:MCMC} <: PMCMCKernel
    "Particle Filter kernel to estimate latent trajectory."
    pf::P
    "MCMC kernel to sample continuous model parameter."
    mcmc::M
    function ParticleMetropolis(pf::P, mcmc::M) where {P<:ParticleFilter,M<:MCMC}
        @argcheck isa(pf.tune.referencing, Marginal) "Cannot use Conditional Filter in Particle Metropolis setting, use Marginal referencing"
        #!NOTE: Only create warning, but let user decide if they want proceed.
        #@argcheck isa(mcmc.kernel, Metropolis) "Only gradient free mcmc kernels allowed as tuning is based on log-target approximation from particle filter instead of objective(val). For gradient based mcmc kernels, use ParticleGibbs instead.")
        if isa(mcmc.tune.stepsize.adaption, BaytesCore.UpdateTrue)
            println("If no objective(val) specified, it is recommended to use a fixed stepsize for the mcmc kernel, as otherwise the kernel will be tuned based on acceptance rate of mcmc kernel.")
        end
        return new{P,M}(pf, mcmc)
    end
end
function ParticleMetropolis(
    filter::F, mcmc::M; kwargs...
) where {F<:ParticleFilterConstructor,M<:MCMCConstructor}
    return PMCMC(ParticleMetropolis, filter, mcmc; kwargs...)
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
    pmcmc::ParticleMetropolis,
    model::ModelWrapper,
    data::D,
    temperature::F = model.info.reconstruct.default.output(1.0),
    update::U=BaytesCore.UpdateTrue(),
) where {D,F<:AbstractFloat,U<:BaytesCore.UpdateBool}
    ## Compute initial logposterior and save initial model value
    ???post??? =
        pmcmc.pf.particles.???objective.cumulative +
        ModelWrappers.log_prior_with_transform(model, pmcmc.mcmc.tune.tagged)
    val = deepcopy(model.val)
    ## Update Objective with new model parameter from other MCMC samplers and/or new/latent data
    objective = Objective(model, data, pmcmc.mcmc.tune.tagged, temperature)
    ## Update Kernel with current objective/configs
    update!(pmcmc.mcmc.kernel, objective, update)
    ## Make MCMC Proposal step
    result???, divergent, accept, sampler_statistic = propagate(
        _rng, pmcmc.mcmc.kernel, pmcmc.mcmc.tune, objective
    )
    ## Make PF proposal step with proposed parameter
    if !divergent
        ModelWrappers.unflatten_constrain!(
            objective.model, pmcmc.mcmc.tune.tagged, result???.?????
        )
    end
    _, pf_diagnostics = propose!(_rng, pmcmc.pf, objective.model, objective.data, temperature, update)
    ## Compute acceptance rate
    ???post?????? =
        pmcmc.pf.particles.???objective.cumulative +
        ModelWrappers.log_prior_with_transform(objective.model, pmcmc.mcmc.tune.tagged)
    accept??? = BaytesCore.AcceptStatistic(_rng, model.info.reconstruct.default.output(???post?????? - ???post???))
    if accept???.accepted
        ## If accepted, update MCMC kernel
        pmcmc.mcmc.kernel.result = result???
    else
        ## Else set model back to initial value
        model.val = val
    end
    #!NOTE: Would need to accept??? here, but makes sampler very noisy. Better to print to either use fixed stepsize or assign objective(val).
    update!(pmcmc.mcmc.tune, pmcmc.mcmc.kernel.result, accept.rate)
    mcmc_diagnostics = MCMCDiagnostics(
        BaytesCore.BaseDiagnostics(
            pmcmc.mcmc.kernel.result.????????,
            objective.temperature,
            predict(_rng, objective),
            pmcmc.mcmc.tune.iter.current
        ),
        sampler_statistic,
        divergent,
        accept???,
        generate(_rng, objective, pmcmc.mcmc.tune.generated),
    )
    ## Assign base diagnostics - ???objective and predictions are taken from particle filter
    diagnostics = BaytesCore.BaseDiagnostics(
        pf_diagnostics.base.???objective, pf_diagnostics.base.temperature, pf_diagnostics.base.prediction,
        mcmc_diagnostics.base.iter
    )
    ## Return pmcmc output
    return model.val, PMCMCDiagnostics(diagnostics, pf_diagnostics, mcmc_diagnostics)
end

############################################################################################
#export
export ParticleMetropolis, propose!
