#= 
    All rights reserved. Ali Mert Ceylan 2020 =#

module AutoDiff

using DiffRules

include("core.jl")
include("interface.jl")

export 
    BD,
    convert,
    promote_rule,
    ndims,
    size,
    getindex,
    setindex!,
    length,
    similar,
    broadcasted,
    +,
    -,
    *,
    /,
    ^,
    sum,
    mean,
    abs2,
    transpose,
    adjoint,
    inv,
    det,
    params,
    @differentiate

end # module
