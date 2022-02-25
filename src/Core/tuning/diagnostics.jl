############################################################################################
"""
$(TYPEDEF)

Contains information about log-likelihood, expected sample size and proposal trajectory.

# Fields
$(TYPEDFIELDS)
"""
struct PMCMCDiagnostics{T,P<:ParticleFilterDiagnostics,M<:MCMCDiagnostics} <: AbstractDiagnostics
    "Diagnostics used for all Baytes kernels"
    base::BaytesCore.BaseDiagnostics{T}
    "Particle Filter diagnostics."
    pf::P
    "MCMC diagnostics."
    mcmc::M
    function PMCMCDiagnostics(
        base::BaytesCore.BaseDiagnostics{T}, pf::P, mcmc::M
    ) where {T,P<:ParticleFilterDiagnostics,M<:MCMCDiagnostics}
        return new{T,P,M}(base, pf, mcmc)
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
#=
function get_prediction(diagnostics::PMCMCDiagnostics)
    return get_prediction(diagnostics.pf)
end
=#
############################################################################################
#export
export PMCMCDiagnostics, generate_showvalues
