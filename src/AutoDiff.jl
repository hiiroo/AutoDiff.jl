#=
    All rights reserved. Ali Mert Ceylan 2020
=#

module AutoDiff

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
    abs2,
    params,
    @differentiate

end # module
