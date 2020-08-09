

function _testint()
    ydy = BD{Int}(1) + BD{Int}(2)
    y, dy = ydy.f
    @test dy(1) == (1, 1)
end

function _testfloat()
    ydy = BD{Float64}(1.0) + BD{Float64}(2.0)
    y, dy = ydy.f
    @test dy(1.0) == (1.0, 1.0)
end

function _testarray()
end
