

using Test

@testset "number_promotions" begin
    @test promote_type(Int, BD{Int}) == BD{Int64}
    @test promote_type(BD{Int}, Int) == BD{Int64}

    @test promote_type(Float64, BD{Float64}) == BD{Float64}
    @test promote_type(BD{Float64}, Float64) == BD{Float64}
end

@testset "wrapped_number_promotions" begin
    @test promote_type(BD{Int}, BD{Float64}) == BD{Float64}
    @test promote_type(BD{Float64}, BD{Int}) == BD{Float64}
end

@testset "wrappedarray_number_promotions" begin
    @test promote_type(BD{Array{Int}}, Int) == Any
    @test promote_type(BD{Int}, Array{Int}) == Any

    @test promote_type(BD{Array{Float64}}, Float64) == Any
    @test promote_type(BD{Float64}, Array{Float64}) == Any

    @test promote_type(BD{Array{Float64}}, Int) == Any
    @test promote_type(Array{Float64}, BD{Int}) == Any

    @test promote_type(BD{Array{Int}}, BD{Float64}) == Any
    @test promote_type(BD{Array{Float64}}, BD{Array{Int}}) == BD{Array{Float64}}
end
