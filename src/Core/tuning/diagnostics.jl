############################################################################################
"""
$(TYPEDEF)

Contains information about log-likelihood, expected sample size and proposal trajectory.

# Fields
$(TYPEDFIELDS)
"""
struct PMCMCDiagnostics{P<:ParticleFilterDiagnostics,M<:MCMCDiagnostics} <: AbstractDiagnostics
    "Particle Filter diagnostics."
    pf::P
    "MCMC diagnostics."
    mcmc::M
    function PMCMCDiagnostics(
        pf::P, mcmc::M
    ) where {P<:ParticleFilterDiagnostics,M<:MCMCDiagnostics}
        return new{P,M}(pf, mcmc)
    end
end

############################################################################################
"""
$(SIGNATURES)
Show relevant diagnostic results.

# Examples
```julia
```

"""
function generate_showvalues(diagnostics::D) where {D<:PMCMCDiagnostics}
    mcmc = generate_showvalues(diagnostics.mcmc)
    pf = generate_showvalues(diagnostics.pf)
    return function showvalues()
        return mcmc()..., pf()...
    end
end

############################################################################################
#export
export PMCMCDiagnostics, generate_showvalues
