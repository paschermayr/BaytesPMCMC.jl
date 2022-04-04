############################################################################################
## Make model
#=
kernel = mcmckernel_pf[1]
references = references_pf[1]
=#

@testset "ParticleMetropolis" begin
    obj = deepcopy(myobjective)
    for kernel in mcmckernel_pf
        mcmcdefault = MCMCDefault(; stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
        mcmckernel = MCMC(
            _rng,
            kernel,
            myobjective_mcmc,
            mcmcdefault
        )
        for references in references_pf
            pfdefault = ParticleFilterDefault(referencing = references,)
            pfkernel = ParticleFilter(
                _rng,
                myobjective_pf,
                pfdefault
            )
            ## Check constructor
            pmcmc_c = ParticleMetropolis(ParticleFilterConstructor(:latent, pfdefault), MCMCConstructor(kernel, (:μ, :σ, :p), mcmcdefault))
            @test keys(obj.model.val) == BaytesPMCMC.get_sym(pmcmc_c)

            mcmcsym = :μ
            PMCMC(ParticleMetropolis, ParticleFilterConstructor(:latent, pfdefault), MCMCConstructor(kernel, mcmcsym, mcmcdefault))
            pmc = ParticleMetropolis(ParticleFilterConstructor(:latent, pfdefault), MCMCConstructor(kernel, mcmcsym, mcmcdefault))
            BaytesPMCMC.get_sym(pmc)
            ## Initialize pmcmc kernel
            pmcmckernel = PMCMC(_rng, ParticleMetropolis, pfkernel, mcmckernel)
            _vals, _diag = propose!(_rng, pmcmckernel, obj.model, obj.data)
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
            results([_diag], pmcmckernel, 2, [.1, .2, .5, .8, .9])
            BaytesPMCMC.result!(pmcmckernel, BaytesPMCMC.get_result(pmcmckernel))
            generate_showvalues(_diag)()

            ## Postprocessing
            @test _diag isa infer(_rng, BaytesPMCMC.AbstractDiagnostics, pmcmckernel, obj.model, obj.data)
            @test _diag.base.prediction isa infer(_rng, pmcmckernel, obj.model, obj.data)
            @test !( predict(_rng, pmcmckernel.kernel.mcmc, obj) isa infer(_rng, pmcmckernel, obj.model, obj.data) )
            @test predict(_rng, pmcmckernel.kernel.pf, obj) isa infer(_rng, pmcmckernel, obj.model, obj.data)
            @test predict(_rng, pmcmckernel.kernel, obj) isa typeof(predict(_rng, pmcmckernel, obj))
            @test _diag.base.ℓobjective == _diag.pf.base.ℓobjective

        end
    end
end

@testset "ParticleGibbs" begin
    obj = deepcopy(myobjective)
    for kernel in mcmckernel_pfa
        mcmcdefault = MCMCDefault(; stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
        mcmckernel = MCMC(
            _rng,
            kernel,
            myobjective_mcmc,
            mcmcdefault
        )
        for references in references_pfa
            pfdefault = ParticleFilterDefault(referencing = references,)
            pfkernel = ParticleFilter(
                _rng,
                myobjective_pf,
                pfdefault
            )
            ## Check constructor
            pmcmc_c = ParticleGibbs(ParticleFilterConstructor(:latent, pfdefault), MCMCConstructor(kernel, (:μ, :σ, :p), mcmcdefault))
            @test keys(obj.model.val) == BaytesPMCMC.get_sym(pmcmc_c)

            mcmcsym = :μ
            PMCMC(ParticleGibbs, ParticleFilterConstructor(:latent, pfdefault), MCMCConstructor(kernel, mcmcsym, mcmcdefault))
            pmc = ParticleGibbs(ParticleFilterConstructor(:latent, pfdefault), MCMCConstructor(kernel, mcmcsym, mcmcdefault))
            BaytesPMCMC.get_sym(pmc)

            ## Initialize pmcmc kernel
            pmcmckernel = PMCMC(_rng, ParticleGibbs, pfkernel, mcmckernel)
            _vals, _diag = propose!(_rng, pmcmckernel, obj.model, obj.data)
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
            results([_diag], pmcmckernel, 2, [.1, .2, .5, .8, .9])
            BaytesPMCMC.result!(pmcmckernel, BaytesPMCMC.get_result(pmcmckernel))
            generate_showvalues(_diag)()

            ## Postprocessing
            @test _diag isa infer(_rng, BaytesPMCMC.AbstractDiagnostics, pmcmckernel, obj.model, obj.data)
            @test _diag.base.prediction isa infer(_rng, pmcmckernel, obj.model, obj.data)
            @test !( predict(_rng, pmcmckernel.kernel.mcmc, obj) isa infer(_rng, pmcmckernel, obj.model, obj.data) )
            @test predict(_rng, pmcmckernel.kernel.pf, obj) isa infer(_rng, pmcmckernel, obj.model, obj.data)
            @test predict(_rng, pmcmckernel.kernel, obj) isa typeof(predict(_rng, pmcmckernel, obj))
            @test _diag.base.ℓobjective == _diag.pf.base.ℓobjective
        end
    end
end
