
#Generalize T, Sum LL function
function general_t_ll(mu,s,nu,x)
    td = TDist(nu)*s + mu
    sum(log.(pdf.(td,x)))
end

#MLE for a Generalize T
function fit_general_t(x)
    global __x
    __x = x
    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    #approximate values based on moments
    start_m = mean(x)
    start_nu = 6.0/kurtosis(x) + 4
    start_s = sqrt(var(x)*(start_nu-2)/start_nu)

    @variable(mle, m, start=start_m)
    @variable(mle, s>=1e-6, start=1)
    @variable(mle, nu>=2.0001, start=start_s)

    #Inner function to abstract away the X value
    function _gtl(mu,s,nu)
        general_t_ll(mu,s,nu,__x)
    end

    register(mle,:tLL,3,_gtl;autodiff=true)
    @NLobjective(
        mle,
        Max,
        tLL(m, s, nu)
    )
    optimize!(mle)

    m = value(m)
    s = value(s)
    nu = value(nu)

    #return the parameters as well as the Distribution Object
    return (m, s, nu, TDist(nu)*s+m)
end


function ewCovar(x,λ)
    m,n = size(x)
    w = Vector{Float64}(undef,m)

    #Remove the mean from the series
    xm = mean.(eachcol(x))
    for j in 1:n
        x[:,j] = x[:,j] .- xm[j]
    end

    #Calculate weight.  Realize we are going from oldest to newest
    for i in 1:m
        w[i] = (1-λ)*λ^(m-i)
    end
    #normalize weights to 1
    w = w ./ sum(w)

    #covariance[i,j] = (w # x)' * x  where # is elementwise multiplication.
    return (w .* x)' * x
end

function simulate_pca(a, nsim; pctExp=1, mean=[],seed=1234)
    n = size(a,1)

    #If the mean is missing then set to 0, otherwise use provided mean
    _mean = fill(0.0,n)
    m = size(mean,1)
    if !isempty(mean)
        copy!(_mean,mean)
    end

    #Eigenvalue decomposition
    vals, vecs = eigen(a)
    vals = real.(vals)
    vecs = real.(vecs)
    #julia returns values lowest to highest, flip them and the vectors
    flip = [i for i in size(vals,1):-1:1]
    vals = vals[flip]
    vecs = vecs[:,flip]
    
    tv = sum(vals)

    posv = findall(x->x>=1e-8,vals)
    if pctExp < 1
        nval = 0
        pct = 0.0
        #figure out how many factors we need for the requested percent explained
        for i in 1:size(posv,1)
            pct += vals[i]/tv
            nval += 1
            if pct >= pctExp 
                break
            end
        end
        if nval < size(posv,1)
            posv = posv[1:nval]
        end
    end
    vals = vals[posv]

    vecs = vecs[:,posv]

    # println("Simulating with $(size(posv,1)) PC Factors: $(sum(vals)/tv*100)% total variance explained")
    B = vecs*diagm(sqrt.(vals))

    Random.seed!(seed)
    m = size(vals,1)
    r = randn(m,nsim)

    out = (B*r)'
    #Loop over itereations and add the mean
    for i in 1:n
        out[:,i] = out[:,i] .+ _mean[i]
    end
    return out
end