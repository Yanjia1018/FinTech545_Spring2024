using LoopVectorization

function colmean(x)
    mean.(eachcol(x))
end

function colmean2(x)
    m = size(x,2)

    out = Vector{Float64}(undef,m)

    for i in 1:m
        out[i] = mean(x[:,i])
    end
    out
end

function colmean3(x)
    m = size(x,2)

    out = Vector{Float64}(undef,m)

    Threads.@threads for i in 1:m
        out[i] = mean(x[:,i])
    end
    out
end

function colmean4(x)
    n,m = size(x)
    out = Vector{Float64}(undef,m)
    out .= 0.0

    @turbo for i in 1:n, j in 1:m
        out[j] += x[i,j]/m
    end

    out
end

function colmean5(x)
    n,m = size(x)
    out = Vector{Float64}(undef,m)

    @turbo for j in 1:m
        s = 0.0
        for i in 1:n
            s += x[i,j]
        end
        out[j] = s/n
    end
    out
end

function demean!(x)
    n,m = size(x)
    xm = colmean5(x)
    @turbo for i in 1:n, j in 1:m
        x[i,j] -= xm[j]
    end
    nothing
end

x = randn((10,2))
colmean(x)
colmean2(x)
colmean3(x)
colmean4(x)
colmean5(x)
x2 = copy(x)
demean!(x2)
colmean(x2)