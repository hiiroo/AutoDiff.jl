

using Test

@testset "broadcasting" begin
    

    @testset "number_to_bdint" begin
        a = BD{Array{Int}}(zeros(Int,3,3))

        y, dy = (1 .+ a).f
        @test dy(1) == (1.0, [1 1 1; 1 1 1; 1 1 1])


        y, dy = (1.0 .+ a).f
        @test dy(1.0) == (1.0, [1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0])
    end

    @testset "number_to_bdfloat" begin
        a = BD{Array{Float64}}(zeros(Float64,3,3))

        y, dy = (1 .+ a).f
        @test dy(1) == (1.0, [1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0])


        y, dy = (1.0 .+ a).f
        @test dy(1.0) == (1.0, [1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0])
    end

    @testset "bdint_to_number" begin
        a = BD{Array{Int}}(zeros(Int,3,3))

        y, dy = (a .+ 1).f
        @test dy(1) == ([1 1 1; 1 1 1; 1 1 1], 1)


        y, dy = (a .+ 1.0).f
        @test dy(1.0) == ([1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0], 1.0)
    end

    @testset "bdfloat_to_number" begin
        a = BD{Array{Float64}}(zeros(Float64,3,3))

        y, dy = (a .+ 1).f
        @test dy(1) == ([1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0], 1.0)


        y, dy = (a .+ 1.0).f
        @test dy(1.0) == ([1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0], 1.0)
    end
end
