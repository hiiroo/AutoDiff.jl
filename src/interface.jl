#=
    All rights reserved. Ali Mert Ceylan 2020
=#

import Base: abs2, sum

function flat(ra)
    if (!(typeof(ra) <: Pair))
        ta = []
        for a in ra
            append!(ta, flat(a))
        end
        return ta
    else 
        return [ra]
    end
end

params(xs) = Dict(flat(xs))

macro differentiate(e)
    args = e.args

    local varname = string(args[1])
    local var = args[1]
    local val = eval(args[2])

    esc(:($var = BD{typeof($val)}($val, Symbol("d$(string($varname))"), )))
end

function sum(B::BD{T}; kwargs...) where T <: AbstractArray
    s = sum([b for b in B.f[1]], kwargs...)
    return BD{typeof(s)}(s, (dy)->(B.f[2](dy.*ones(size(B)))))
end

abs2(b::BD{T}) where T <: AbstractArray = abs2(b.f[1], (dy)->(b.f[2](dy.*2b.f[1])))
broadcasted(::typeof(abs2), b::BD{T}) where T <: AbstractArray = BD{T}(abs2.(b.f[1]), (dy)->(b.f[2](dy.*2b.f[1])))
