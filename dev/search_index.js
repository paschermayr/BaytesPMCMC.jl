var documenterSearchIndex = {"docs":
[{"location":"intro/#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"intro/","page":"Introduction","title":"Introduction","text":"Yet to be properly done.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = BaytesPMCMC","category":"page"},{"location":"#BaytesPMCMC","page":"Home","title":"BaytesPMCMC","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for BaytesPMCMC.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [BaytesPMCMC]","category":"page"},{"location":"#BaytesPMCMC.PMCMC","page":"Home","title":"BaytesPMCMC.PMCMC","text":"struct PMCMC{M<:PMCMCKernel, N<:PMCMCTune} <: BaytesCore.AbstractAlgorithm\n\nParticle MCMC container, container kernel and tuning.\n\nFields\n\nkernel::PMCMCKernel\nPMCMC sampler\ntune::PMCMCTune\nTuning configuration for kernel.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesPMCMC.PMCMCConstructor","page":"Home","title":"BaytesPMCMC.PMCMCConstructor","text":"Callable struct to make initializing PMCMC sampler easier in sampling library.\n\nExamples\n\n\n\n\n\n\n\n","category":"type"},{"location":"#BaytesPMCMC.PMCMCDefault","page":"Home","title":"BaytesPMCMC.PMCMCDefault","text":"struct PMCMCDefault\n\nDefault arguments for PMCMC constructor.\n\nFields\n\n\n\n\n\n","category":"type"},{"location":"#BaytesPMCMC.PMCMCDiagnostics","page":"Home","title":"BaytesPMCMC.PMCMCDiagnostics","text":"struct PMCMCDiagnostics{P<:BaytesFilters.ParticleFilterDiagnostics, M<:BaytesMCMC.MCMCDiagnostics} <: BaytesCore.AbstractDiagnostics\n\nContains information about log-likelihood, expected sample size and proposal trajectory.\n\nFields\n\npf::BaytesFilters.ParticleFilterDiagnostics\nParticle Filter diagnostics.\nmcmc::BaytesMCMC.MCMCDiagnostics\nMCMC diagnostics.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesPMCMC.PMCMCTune","page":"Home","title":"BaytesPMCMC.PMCMCTune","text":"struct PMCMCTune <: BaytesCore.AbstractTune\n\nPMCMC tuning container.\n\nFields\n\n\n\n\n\n","category":"type"},{"location":"#BaytesPMCMC.ParticleGibbs","page":"Home","title":"BaytesPMCMC.ParticleGibbs","text":"struct ParticleGibbs{P<:BaytesFilters.ParticleFilter, M<:BaytesMCMC.MCMC} <: PMCMCKernel\n\nDefault arguments for PMCMC constructor.\n\nFields\n\npf::BaytesFilters.ParticleFilter\nParticle Filter kernel to estimate latent trajectory.\nmcmc::BaytesMCMC.MCMC\nMCMC kernel to sample continuous model parameter.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesPMCMC.ParticleMetropolis","page":"Home","title":"BaytesPMCMC.ParticleMetropolis","text":"struct ParticleMetropolis{P<:BaytesFilters.ParticleFilter, M<:BaytesMCMC.MCMC} <: PMCMCKernel\n\nDefault arguments for PMCMC constructor.\n\nFields\n\npf::BaytesFilters.ParticleFilter\nParticle Filter kernel to estimate latent trajectory.\nmcmc::BaytesMCMC.MCMC\nMCMC kernel to sample continuous model parameter.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesCore.generate_showvalues-Tuple{D} where D<:PMCMCDiagnostics","page":"Home","title":"BaytesCore.generate_showvalues","text":"generate_showvalues(diagnostics)\n\n\nShow relevant diagnostic results.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.infer-Union{Tuple{D}, Tuple{Random.AbstractRNG, PMCMC, ModelWrappers.ModelWrapper, D}, Tuple{Random.AbstractRNG, PMCMC, ModelWrappers.ModelWrapper, D, Bool}} where D","page":"Home","title":"BaytesCore.infer","text":"infer(_rng, pmcmc, model, data)\ninfer(_rng, pmcmc, model, data, alldata)\n\n\nInfer type of predictions of kernel.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.infer-Union{Tuple{D}, Tuple{Random.AbstractRNG, Type{BaytesCore.AbstractDiagnostics}, PMCMC, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesCore.infer","text":"infer(_rng, diagnostics, pmcmc, model, data)\n\n\nInfer PMCMC diagnostics type.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose!-Union{Tuple{U}, Tuple{D}, Tuple{Random.AbstractRNG, PMCMC, ModelWrappers.ModelWrapper, D}, Tuple{Random.AbstractRNG, PMCMC, ModelWrappers.ModelWrapper, D, U}} where {D, U<:UpdateBool}","page":"Home","title":"BaytesCore.propose!","text":"propose!(_rng, pmcmc, model, data)\npropose!(_rng, pmcmc, model, data, update)\n\n\nPropose new parameter with pmcmc sampler. If update=true, objective function will be updated with input model and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose!-Union{Tuple{U}, Tuple{D}, Tuple{Random.AbstractRNG, ParticleGibbs, ModelWrappers.ModelWrapper, D}, Tuple{Random.AbstractRNG, ParticleGibbs, ModelWrappers.ModelWrapper, D, U}} where {D, U<:UpdateBool}","page":"Home","title":"BaytesCore.propose!","text":"propose!(_rng, pmcmc, model, data)\npropose!(_rng, pmcmc, model, data, update)\n\n\nPropose new parameter with mcmc psampler. If update=true, objective function will be updated with input model and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose!-Union{Tuple{U}, Tuple{D}, Tuple{Random.AbstractRNG, ParticleMetropolis, ModelWrappers.ModelWrapper, D}, Tuple{Random.AbstractRNG, ParticleMetropolis, ModelWrappers.ModelWrapper, D, U}} where {D, U<:UpdateBool}","page":"Home","title":"BaytesCore.propose!","text":"propose!(_rng, pmcmc, model, data)\npropose!(_rng, pmcmc, model, data, update)\n\n\nPropose new parameter with pmcmc sampler. If update=true, objective function will be updated with input model and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"}]
}
