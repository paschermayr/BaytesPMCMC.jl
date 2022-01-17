############################################################################################
## Make model
@testset "ParticleMetropolis" begin
    obj = deepcopy(myobjective)
    for kernel in mcmckernel_pf
        mcmckernel = MCMC(kernel, myobjective_mcmc;
            default = MCMCDefault(; config_kw = (;stepsizeadaption = UpdateFalse()))
        )
        for references in references_pf
            pfkernel = ParticleFilter(myobjective_pf ;
                default = ParticleFilterDefault(referencing = references,)
            )
            ## Initialize pmcmc kernel
            pmcmckernel = PMCMC(ParticleMetropolis, pfkernel, mcmckernel)
            propose!(_rng, pmcmckernel, obj.model, obj.data)
            ## Check if constructor can be used
            pfconstructor = BaytesFilters.ParticleFilterConstructor(
                :latent,
                ParticleFilterDefault(referencing = references,)
            )
            mcmcconstructor = BaytesMCMC.MCMCConstructor(
                kernel,
                keys(myobjective_mcmc.tagged.parameter),
                MCMCDefault(; config_kw = (;stepsizeadaption = UpdateFalse()))
            )
            constructor = PMCMCConstructor(
                ParticleMetropolis,
                pfconstructor,
                mcmcconstructor,
                PMCMCDefault()
            )
            constructor(_rng, obj.model, obj.data, 1, TemperDefault())
        end
    end
end

@testset "ParticleGibbs" begin
    obj = deepcopy(myobjective)
    for kernel in mcmckernel_pfa
        mcmckernel = MCMC(kernel, myobjective_mcmc;
            default = MCMCDefault(; config_kw = (;stepsizeadaption = UpdateFalse()))
        )
        for references in references_pfa
            pfkernel = ParticleFilter(myobjective_pf ;
                default = ParticleFilterDefault(referencing = references,)
            )
            ## Initialize pmcmc kernel
            pmcmckernel = PMCMC(ParticleGibbs, pfkernel, mcmckernel)
            propose!(_rng, pmcmckernel, obj.model, obj.data)
            ## Check if constructor can be used
            pfconstructor = BaytesFilters.ParticleFilterConstructor(
                :latent,
                ParticleFilterDefault(referencing = references,)
            )
            mcmcconstructor = BaytesMCMC.MCMCConstructor(
                kernel,
                keys(myobjective_mcmc.tagged.parameter),
                MCMCDefault(; config_kw = (;stepsizeadaption = UpdateFalse()))
            )
            constructor = PMCMCConstructor(
                ParticleGibbs,
                pfconstructor,
                mcmcconstructor,
                PMCMCDefault()
            )
            constructor(_rng, obj.model, obj.data, 1, TemperDefault())
        end
    end
end
