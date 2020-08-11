

using Test

@testset "broadcasting" begin
    
    a = BD{Array{Int}}(zeros(Int,3,3))

    @testset "number_to_bdnumber" begin
        y, dy = (1 .+ a).f
        @test dy(1) == (1.0, [1 1 1; 1 1 1; 1 1 1])
    end
end

# int bdarrayint
# int bdarrayfloat
# float bdarrayfloat
# float bdarrayint

# bdarrayint int
# bdarrayfloat int
# bdarrayfloat float
# bdarrayint float


