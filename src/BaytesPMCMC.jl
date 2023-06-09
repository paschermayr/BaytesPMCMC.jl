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
    AcceptStatistic,
    SampleDefault,
    ProposalTune

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
    get_sym,
    generate_showvalues,
    generate,
    UpdateBool,
    UpdateTrue,
    UpdateFalse

using ModelWrappers:
    ModelWrappers,
    ModelWrapper,
    Tagged,
    Objective,
    dynamics,
    _checkprior

import ModelWrappers: ModelWrappers, predict

using BaytesDiff:
    BaytesDiff,
    DiffObjective,
    ℓObjectiveResult


using BaytesMCMC, BaytesFilters

using DocStringExtensions:
    DocStringExtensions, TYPEDEF, TYPEDFIELDS, FIELDS, SIGNATURES, FUNCTIONNAME
using ArgCheck: ArgCheck, @argcheck
using SimpleUnPack: SimpleUnPack, @unpack, @pack!

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
    generate,

    PMCMCKernel,
    update!,
    init,
    init!,
    propose,
    propose!,
    propagate,
    propagate!,
    SampleDefault,

    #ModelWrappers
    dynamics,
    predict
end
