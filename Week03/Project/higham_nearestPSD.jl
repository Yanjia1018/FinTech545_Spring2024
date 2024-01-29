function higham_nearestPSD(pc,epsilon=1e-9,maxIter=100,tol=1e-9)

    n = size(pc,1)
    W = diagm(fill(1.0,n))

    deltaS = 0

    Yk = copy(pc)
    norml = typemax(Float64)
    i=1

    while i <= maxIter
        # println("$i - $norml")
        Rk = Yk .- deltaS
        #Ps Update
        Xk = _getPS(Rk,W)
        deltaS = Xk - Rk
        #Pu Update
        Yk = _getPu(Xk,W)
        #Get Norm
        norm = wgtNorm(Yk-pc,W)
        #Smallest Eigenvalue
        minEigVal = min(real.(eigvals(Yk))...)

        if norm - norml < tol && minEigVal > -epsilon
            # Norm converged and matrix is at least PSD
            break
        end
        # println("$norml -> $norm")
        norml = norm
        i += 1
    end
    if i < maxIter 
        println("Converged in $i iterations.")
    else
        println("Convergence failed after $(i-1) iterations")
    end
    return Yk
end