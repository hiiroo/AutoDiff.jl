

using Test

@testset "arithmetic" begin

    @testset "integer" begin
        y, dy = (BD{Int}(1) + BD{Int}(2)).f
        @test dy(1) == (1, 1)

        y, dy = (BD{Int}(1) - BD{Int}(2)).f
        @test dy(1) == (1, -1)

        y, dy = (BD{Int}(1) * BD{Int}(2)).f
        @test dy(1) == (2, 1)

        y, dy = (BD{Int}(1) / BD{Int}(2)).f
        @test dy(1) == (0.5, -0.25)

        y, dy = (BD{Int}(1) ^ BD{Int}(2)).f
        @test dy(1) == (2, 0.0)
    end

    @testset "floatingpoint" begin
        y, dy = (BD{Float64}(1.0) + BD{Float64}(2.0)).f
        @test dy(1.0) == (1.0, 1.0)

        y, dy = (BD{Float64}(1.0) - BD{Float64}(2.0)).f
        @test dy(1.0) == (1.0, -1.0)

        y, dy = (BD{Float64}(1.0) * BD{Float64}(2.0)).f
        @test dy(1.0) == (2.0, 1.0)

        y, dy = (BD{Float64}(1.0) / BD{Float64}(2.0)).f
        @test dy(1.0) == (0.5, -0.25)

        y, dy = (BD{Float64}(1.0) ^ BD{Float64}(2.0)).f
        @test dy(1.0) == (2.0, 0.0)
    end
    
    @testset "intarray" begin
        a, b = BD{Array{Int}}(zeros(Int, 3, 3).+1), BD{Array{Int}}(zeros(Int, 3, 3).+2)

        y, dy = (a + b).f
        @test dy(ones(Int, 3, 3)) == ([1 1 1; 1 1 1; 1 1 1], [1 1 1; 1 1 1; 1 1 1])

        y, dy = (a - b).f
        @test dy(ones(Int, 3, 3)) == ([1 1 1; 1 1 1; 1 1 1], [-1 -1 -1; -1 -1 -1; -1 -1 -1])

        y, dy = (a * b).f
        @test dy(ones(Int, 3, 3)) == ([6 6 6; 6 6 6; 6 6 6], [3 3 3; 3 3 3; 3 3 3])     
    end

    @testset "floatarray" begin
        a, b = BD{Array{Float64}}(zeros(Float64, 3, 3).+1), BD{Array{Float64}}(zeros(Float64, 3, 3).+2)

        y, dy = (a + b).f
        @test dy(ones(Float64, 3, 3)) == ([1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0], [1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0])

        y, dy = (a - b).f
        @test dy(ones(Float64, 3, 3)) == ([1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0], [-1.0 -1.0 -1.0; -1.0 -1.0 -1.0; -1.0 -1.0 -1.0])

        y, dy = (a * b).f
        @test dy(ones(Float64, 3, 3)) == ([6.0 6.0 6.0; 6.0 6.0 6.0; 6.0 6.0 6.0], [3.0 3.0 3.0; 3.0 3.0 3.0; 3.0 3.0 3.0])     
    end


end
