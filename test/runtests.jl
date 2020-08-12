#=
    All rights reserved. Ali Mert Ceylan 2020
=#

using Test
using AutoDiff


@testset "AutoDiff" begin

@time include("arthmtc.jl")    
@time include("promtn.jl")
@time include("brdcstng.jl")
# @time include("specfncs.jl")

end
