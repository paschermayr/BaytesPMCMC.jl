############################################################################################
module BaytesPMCMC

############################################################################################
#Import modules
using BaytesCore:
    BaytesCore,
    AbstractAlgorithm,
    AbstractTune,
    AbstractConfiguration,
    AbstractDiagnostics,
    AbstractKernel,
    AbstractConstructor,
    update,
    AcceptStatistic

import BaytesCore:
    BaytesCore,
    update!,
    infer,
    results,
    init,
    init!,
    propose,
    propose!,
    propagate,
    propagate!,
    result!,
    get_result,
    get_ℓweight,
    get_prediction,
    get_tagged,
    get_sym,
    generate_showvalues,
    UpdateBool,
    UpdateTrue,
    UpdateFalse

using ModelWrappers:
    ModelWrappers,
    ModelWrapper,
    Tagged,
    Objective,
    DiffObjective,
    ℓObjectiveResult,
    dynamics,
    predict,
    generate,
    _checkprior

using BaytesMCMC, BaytesFilters

using DocStringExtensions:
    DocStringExtensions, TYPEDEF, TYPEDFIELDS, FIELDS, SIGNATURES, FUNCTIONNAME
using ArgCheck: ArgCheck, @argcheck
using UnPack: UnPack, @unpack, @pack!

using Random: Random, AbstractRNG, GLOBAL_RNG

############################################################################################
# Define abstract types
abstract type PMCMCKernel <: AbstractKernel end

############################################################################################
# Import sub-folder
include("Core/Core.jl")
include("Kernels/Kernels.jl")

############################################################################################
export
    #BaytesCore
    UpdateBool,
    UpdateTrue,
    UpdateFalse,

    PMCMCKernel,
    update!,
    init,
    init!,
    propose,
    propose!,
    propagate,
    propagate!,

    #ModelWrappers
    dynamics,
    predict,
    generate
end
