#= 
    All rights reserved. Ali Mert Ceylan 2020 =#

import Base: broadcasted, abs2, sum, inv
import Statistics: mean
import LinearAlgebra: adjoint, transpose, det

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
    local val = args[2]

    esc(:($var = BD{typeof($val)}($val, Symbol("d$(string($varname))"), )))
end

function sum(B::BD{T}; kwargs...) where T <: AbstractArray
    s = sum([b for b in B.f[1]], kwargs...)
    return BD{typeof(s)}(s, (dy)->(B.f[2](dy .* ones(size(B)))))
end

# @primitive mean(x;d...),dy  (dy.*one.(x)./(length(x)÷length(dy)))
function mean(bd::BD{T}; kwargs...) where T <: AbstractArray
    s = mean(bd.f[1])
    return BD{typeof(s)}(s, (dy)->(bd.f[2](dy .* ones(size(bd.f[1])) ./ length(bd.f[1]))))# dy.*ones(size(bd.f[1]))./length(bd.f[1])
end

abs2(b::BD{T}) where T <: Number = BD{T}(abs2(b.f[1]), (dy)->(b.f[2](dy * 2b.f[1])))
broadcasted(::typeof(abs2), b::BD{T}) where T <: AbstractArray = BD{T}(abs2.(b.f[1]), (dy)->(b.f[2](dy .* 2b.f[1])))

transpose(b::BD{T}) where T <: AbstractArray = BD{T}(transpose(b.f[1]), (dy)->(transpose(dy)))
adjoint(b::BD{T}) where T <: AbstractArray = BD{T}(adjoint(b.f[1]), (dy)->(adjoint(dy)))

# inv(x),dy,y  (yt=y'; -yt*dy*yt) # addtest(:inv, rand(2,2)) # ref https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf 
inv(b::BD{T}) where T <: AbstractArray = BD{T}(inv(b.f[1]), (dy)->(yt = inv(b.f[1])'; -yt * dy * yt))

# det(x),dy,y  dy*y*inv(x)'
det(b::BD{T}) where T <: AbstractArray = BD{T}(det(b.f[1]), (dy)->(dy * det(b.f[1]) * inv(b.f[1])'))

# TODO diag(x),dy,y   diagm(0=>dy)  # alternative: Diagonal(dy)
# TODO diag(x,i),dy,y   diagm(i=>dy)  # warning: these only works for square matrices
# TODO dot(x1, x2),dy,y  dy*x2  dy*x1  # addtestN(:dot, rand(3,2), rand(3,2))

# TODO @zerograd axes(x,i...)
# TODO @primitive maximum(x;d...),dy,y  (dy.*(y.==x))
# TODO @primitive maximum(f::typeof(abs),x;d...),dy,y  nothing  (dy.*(y.==abs.(x)).*sign.(x))
# TODO @primitive minimum(x;d...),dy,y  (dy.*(y.==x))
# TODO @primitive minimum(f::typeof(abs),x;d...),dy,y  nothing  (dy.*(y.==abs.(x)).*sign.(x))
