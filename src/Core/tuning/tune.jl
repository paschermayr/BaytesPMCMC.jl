############################################################################################
"""
$(TYPEDEF)

PMCMC tuning container.

# Fields
$(TYPEDFIELDS)
"""
struct PMCMCTune{T<:Tagged} <: AbstractTune
    "Tagged Model parameter."
    tagged::T
    function PMCMCTune(
        tagged::T
    ) where {T<:Tagged}
        return new{T}(tagged)
    end
end

############################################################################################
#export
export PMCMCTune
