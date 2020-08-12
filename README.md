# AutoDiff

AutoDiff.jl is a basic reverse differentiation package inspired by the talk of Prof. Edelman. Similar to DualNumbers, AutoDiff provides the type BD. BD can be considered as back-derivation. After every operation, a tuple of the result of the operation and the reverse function which will provide gradients w.r.t inputs of the operation.

```julia
using AutoDiff

f(x) = x^2

@differentiate a = 3

yfdy = f(a)
y,fdy=yfdy.f

println(y) # (:da => 6, 2.1972245773362196), since only a is tracked, :da is stored as a pair
println(params(fdy(1)))# That will give a dictionary of tracked variables, for this case :da => 6
```

Any type which is not tracked will be automatically converted to a BD type for derivative calculation. Tracked variables can be obtained easily by calling derivative function inside the params function.

Slightly more complicated example:

```julia
using AutoDiff

@differentiate i = rand(4, 1)
@differentiate w = rand(4, 4)
@differentiate o = rand(3, 4)

function f(x)
    x = w*x
    x = o*x
    return x
end

y = sum(abs2.(1.0 .-f(i)))

_dy, _rest = y.f[2](y.f[1])

grads(rest)
```

You may check the example folder, there are jupyter notebooks with basic usage.

