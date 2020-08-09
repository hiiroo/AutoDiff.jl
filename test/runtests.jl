#=
    All rights reserved. Ali Mert Ceylan 2020
=#

@testset "AutoDiff" begin

    @time include("promtn.jl")
    @time include("arthmtc.jl")
    @time include("specops.jl")
    @time include("brdcstng.jl")
    @time include("specfncs.jl")

end
