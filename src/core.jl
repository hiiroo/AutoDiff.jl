#=
    All rights reserved. Ali Mert Ceylan 2020
=#

import Base: +, /, -, *, convert, promote_rule, length, size, ndims, getindex, setindex!, similar, broadcasted, broadcastable

"""
    unbroadcast(x,dx)

Bring dx to x's size via unbroadcasting (reduction). This is needed
when defining gradients of multi-argument broadcasting functions where
the arguments and the result may be of different sizes.

Courtesy of AutoGrad.jl
"""
function unbroadcast(x, dx)
    if size(x)==size(dx)
        return dx
    elseif isa(x,Number)
        return sum(dx)
    elseif isa(dx,Number)
        return fill!(similar(x),dx)
    else
        d = []
        for i=1:ndims(dx)
            size(x,i) == size(dx,i) > 1 && continue
            size(x,i) != 1 && throw(DimensionMismatch())
            push!(d,i)
        end
        length(d)==1 && (d=d[1])
        return reshape(sum(dx, dims=d), size(x))
    end
end

# This fixes unbroadcast when x isa Broadcasted
Base.size(x::Base.Broadcast.Broadcasted,i...)=length.(axes(x,i...))

mutable struct BD{V}
    f::Tuple{V, Function}
end

function BD{V}(v1::V) where V <: Union{Number, AbstractArray}
    return BD{V}((v1, dy->dy))
end

function BD{V}(v1::V, s::Symbol) where V <: Union{Number, AbstractArray}
    return BD{V}((v1, dy->Pair(s, dy)))
end

function BD{V}(v1::V, v2) where V <: Union{Number, AbstractArray}
    return BD{V}((v1, v2))
end

+(x::BD{T}, y::BD{T}) where T <: Union{Number, AbstractArray} = BD{T}(x.f[1] + y.f[1], (dy)->(x.f[2](dy), y.f[2](dy)))
-(x::BD{T}, y::BD{T}) where T <: Union{Number, AbstractArray} = BD{T}(x.f[1] - y.f[1], (dy)->(x.f[2](dy), y.f[2](dy)))
*(x::BD{T}, y::BD{T}) where T <: Union{Number, AbstractArray} = BD{T}(x.f[1] * y.f[1], (dy)->(x.f[2](dy*y.f[1]'), y.f[2](x.f[1]'*dy)))
/(x::BD{T}, y::BD{T}) where T <: Number = BD{T}(x.f[1] / y.f[1], (dy)->(x.f[2](dy/y.f[1]), y.f[2]((-dy*x.f[1])/(y.f[2]^2))))

convert(::Type{BD}, x::T) where T <: Union{Number, AbstractArray} = BD{T}(x)
promote_rule(::Type{BD}, ::Type{T}) where T <: Number = BD{T}
ndims(bd::BD) = ndims(bd.f[1])
size(bd::BD) = size(bd.f[1])
getindex(bd::BD, i) = getindex(bd.f[1], i)
setindex!(bd::BD, i) = setindex!(bd.f[1], i)

length(bd::BD{T}) where T <: AbstractArray = length(bd.f[1])
similar(bd::BD{T}) where T <: Number = BD{eltype(bd)}(0)
similar(bd::BD{T}) where T <: AbstractArray = BD{eltype(bd)}(zeros(size(bd)))

broadcasted(::typeof(+), x::BD{T}, y::BD{U}) where {T <: Number, U <: AbstractArray} = BD{U}(x.f[1].+y.f[1], (dy)->(x.f[2](unbroadcast(x.f[1], dy)),  y.f[2](unbroadcast(y.f[1], dy))))
broadcasted(::typeof(+), x::BD{U}, y::BD{T}) where {T <: Number, U <: AbstractArray} = BD{U}(x.f[1].+y.f[1], (dy)->(x.f[2](unbroadcast(x.f[1], dy)),  y.f[2](unbroadcast(y.f[1], dy))))

broadcasted(::typeof(-), x::BD{T}, y::BD{U}) where {T <: Number, U <: AbstractArray} = BD{U}(x.f[1].-y.f[1], (dy)->(x.f[2](unbroadcast(x.f[1], dy)),  y.f[2](unbroadcast(y.f[1], -dy))))
broadcasted(::typeof(-), x::BD{U}, y::BD{T}) where {T <: Number, U <: AbstractArray} = BD{U}(x.f[1].-y.f[1], (dy)->(x.f[2](unbroadcast(x.f[1], dy)),  y.f[2](unbroadcast(y.f[1], -dy))))

broadcasted(::typeof(*), x::BD{T}, y::BD{U}) where {T <: Number, U <: AbstractArray} = BD{U}(x.f[1].*y.f[1], (dy)->(x.f[2](unbroadcast(x.f[1], dy.*y.f[1])),  y.f[2](unbroadcast(y.f[1], x.f[1].*dy))))
broadcasted(::typeof(*), x::BD{U}, y::BD{T}) where {T <: Number, U <: AbstractArray} = BD{U}(x.f[1].*y.f[1], (dy)->(x.f[2](unbroadcast(x.f[1], dy.*y.f[1])),  y.f[2](unbroadcast(y.f[1], x.f[1].*dy))))

broadcasted(::typeof(/), x::BD{T}, y::BD{U}) where {T <: Number, U <: AbstractArray} = BD{U}(x.f[1]./y.f[1], (dy)->(x.f[2](unbroadcast(x.f[1], dy./y.f[1])),  y.f[2](unbroadcast(y.f[1], (-dy.*x.f[1])./abs2.(y.f[2])))))
broadcasted(::typeof(/), x::BD{U}, y::BD{T}) where {T <: Number, U <: AbstractArray} = BD{U}(x.f[1]./y.f[1], (dy)->(x.f[2](unbroadcast(x.f[1], dy./y.f[1])),  y.f[2](unbroadcast(y.f[1], (-dy.*x.f[1])./abs2.(y.f[2])))))
