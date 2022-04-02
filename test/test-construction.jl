############################################################################################
## Make model
@testset "ParticleMetropolis" begin
    obj = deepcopy(myobjective)
    for kernel in mcmckernel_pf
        mcmckernel = MCMC(
            _rng,
            kernel,
            myobjective_mcmc,
            MCMCDefault(; stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
        )
        for references in references_pf
            pfkernel = ParticleFilter(
                _rng,
                myobjective_pf,
                ParticleFilterDefault(referencing = references,)
            )
            ## Initialize pmcmc kernel
            pmcmckernel = PMCMC(_rng, ParticleMetropolis, pfkernel, mcmckernel)
            propose!(_rng, pmcmckernel, obj.model, obj.data)
            ## Check if constructor can be used
            pfconstructor = BaytesFilters.ParticleFilterConstructor(
                :latent,
                ParticleFilterDefault(referencing = references,)
            )
            mcmcconstructor = BaytesMCMC.MCMCConstructor(
                kernel,
                keys(myobjective_mcmc.tagged.parameter),
                MCMCDefault(; stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
            )
            constructor = PMCMCConstructor(
                ParticleMetropolis,
                pfconstructor,
                mcmcconstructor,
                PMCMCDefault()
            )
            constructor(_rng, obj.model, obj.data, 1.0, SampleDefault())

            #!NOTE: Placeholder
            @test 1 == 1
        end
    end
end

@testset "ParticleGibbs" begin
    obj = deepcopy(myobjective)
    for kernel in mcmckernel_pfa
        mcmckernel = MCMC(
            _rng,
            kernel,
            myobjective_mcmc,
            MCMCDefault(; stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
        )
        for references in references_pfa
            pfkernel = ParticleFilter(
                _rng,
                myobjective_pf,
                ParticleFilterDefault(referencing = references,)
            )
            ## Initialize pmcmc kernel
            pmcmckernel = PMCMC(_rng, ParticleGibbs, pfkernel, mcmckernel)
            propose!(_rng, pmcmckernel, obj.model, obj.data)
            ## Check if constructor can be used
            pfconstructor = BaytesFilters.ParticleFilterConstructor(
                :latent,
                ParticleFilterDefault(referencing = references,)
            )
            mcmcconstructor = BaytesMCMC.MCMCConstructor(
                kernel,
                keys(myobjective_mcmc.tagged.parameter),
                MCMCDefault(; stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
            )
            constructor = PMCMCConstructor(
                ParticleGibbs,
                pfconstructor,
                mcmcconstructor,
                PMCMCDefault()
            )
            constructor(_rng, obj.model, obj.data, 1., SampleDefault())

            #!NOTE: Placeholder
            @test 1 == 1
        end
    end
end
