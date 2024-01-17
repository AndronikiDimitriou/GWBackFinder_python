
function f26(S, S0, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22, n23, n24, n25, As, Sb1, Sb2, Sb3, Sb4, Sb5, Sb6, Sb7, Sb8, Sb9, Sb10, Sb11, Sb12, Sb13, Sb14, Sb15, Sb16, Sb17, Sb18, Sb19, Sb20, Sb21, Sb22, Sb23, Sb24, Sb25)
    """
    Function that calculates a value based on a piecewise-powerlaw expression.

    Parameters:
    - S: Input value.
    - S0: Reference value.
    - n0 to n25: Exponents for various terms.
    - As: Scaling factor.
    - Sb1 to Sb25: Threshold values for different ranges.

    Returns:
    - result: Calculated value based on the specified conditions.
    """
   
    if S <= Sb1
        result = As .* (Sb12 / S0).^(n12 .- n11) .* (Sb11 / S0).^(n11 .- n10) .* (Sb10 / S0).^(n10 .- n9) .* (Sb9 / S0).^(n9 .- n8).*(Sb8 / S0).^(n8 .- n7) .* (Sb7 / S0).^(n7 .- n6) .* (Sb6 / S0).^(n6 .- n5) .* (Sb5 / S0).^(n5 .- n4) .* (Sb4 / S0).^(n4 .- n3).*(Sb3 / S0).^(n3 .- n2) .* (Sb2 / S0).^(n2 .- n1) .* (Sb1 / S0).^(n1 .- n0) .* (S / S0).^(n0)

    elseif Sb1 < S <= Sb2
        result = As .* (Sb12 / S0).^(n12 .- n11) .* (Sb11 / S0).^(n11 .- n10) .* (Sb10 / S0).^(n10 .- n9) .* (Sb9 / S0).^(n9 .- n8).*(Sb8 / S0).^(n8 .- n7) .* (Sb7 / S0).^(n7 .- n6) .* (Sb6 / S0).^(n6 .- n5) .* (Sb5 / S0).^(n5 .- n4) .* (Sb4 / S0).^(n4 .- n3).*(Sb3 / S0).^(n3 .- n2) .* (Sb2 / S0).^(n2 .- n1) .* (S / S0).^(n1)

    elseif Sb2 < S <= Sb3
        result = As .* (Sb12 / S0).^(n12 .- n11) .* (Sb11 / S0).^(n11 .- n10) .* (Sb10 / S0).^(n10 .- n9) .* (Sb9 / S0).^(n9 .- n8).*(Sb8 / S0).^(n8 .- n7) .* (Sb7 / S0).^(n7 .- n6) .* (Sb6 / S0).^(n6 .- n5) .* (Sb5 / S0).^(n5 .- n4) .* (Sb4 / S0).^(n4 .- n3).*(Sb3 / S0).^(n3 .- n2) .* (S / S0).^(n2)

    elseif Sb3 < S <= Sb4
        result = As .* (Sb12 / S0).^(n12 .- n11) .* (Sb11 / S0).^(n11 .- n10) .* (Sb10 / S0).^(n10 .- n9) .* (Sb9 / S0).^(n9 .- n8).*(Sb8 / S0).^(n8 .- n7) .* (Sb7 / S0).^(n7 .- n6) .* (Sb6 / S0).^(n6 .- n5) .* (Sb5 / S0).^(n5 .- n4) .* (Sb4 / S0).^(n4 .- n3).*(S / S0).^(n3)

    elseif Sb4 < S <= Sb5
        result = As .* (Sb12 / S0).^(n12 .- n11) .* (Sb11 / S0).^(n11 .- n10) .* (Sb10 / S0).^(n10 .- n9) .* (Sb9 / S0).^(n9 .- n8).*(Sb8 / S0).^(n8 .- n7) .* (Sb7 / S0).^(n7 .- n6) .* (Sb6 / S0).^(n6 .- n5) .* (Sb5 / S0).^(n5 .- n4) .* (S / S0).^(n4)

    elseif Sb5 < S <= Sb6
        result = As .* (Sb12 / S0).^(n12 .- n11) .* (Sb11 / S0).^(n11 .- n10) .* (Sb10 / S0).^(n10 .- n9) .* (Sb9 / S0).^(n9 .- n8).*(Sb8 / S0).^(n8 .- n7) .* (Sb7 / S0).^(n7 .- n6) .* (Sb6 / S0).^(n6 .- n5) .* (S / S0).^(n5)

    elseif Sb6 < S <= Sb7
        result = As .* (Sb12 / S0).^(n12 .- n11) .* (Sb11 / S0).^(n11 .- n10) .* (Sb10 / S0).^(n10 .- n9) .* (Sb9 / S0).^(n9 .- n8).*(Sb8 / S0).^(n8 .- n7) .* (Sb7 / S0).^(n7 .- n6) .* (S / S0).^(n6)

    elseif Sb7 < S <= Sb8
        result = As .* (Sb12 / S0).^(n12 .- n11) .* (Sb11 / S0).^(n11 .- n10) .* (Sb10 / S0).^(n10 .- n9) .* (Sb9 / S0).^(n9 .- n8).*(Sb8 / S0).^(n8 .- n7) .* (S / S0).^(n7)

    elseif Sb8 < S <= Sb9
        result = As .* (Sb12 / S0).^(n12 .- n11) .* (Sb11 / S0).^(n11 .- n10) .* (Sb10 / S0).^(n10 .- n9) .* (Sb9 / S0).^(n9 .- n8).*(S / S0).^(n8)

    elseif Sb9 < S <= Sb10
        result = As .* (Sb12 / S0).^(n12 .- n11) .* (Sb11 / S0).^(n11 .- n10) .* (Sb10 / S0).^(n10.- n9) .* (S / S0).^(n9)

    elseif Sb10 < S <= Sb11
        result = As .* (Sb12 / S0).^(n12 .- n11) .* (Sb11 / S0).^(n11 .- n10) .* (S / S0).^(n10)

    elseif Sb11 < S <= Sb12
        result = As .* (Sb12 / S0).^(n12 .- n11) .* (S / S0).^(n11)

    elseif Sb12 < S <= Sb13
        result = As .* (S / S0).^(n12)

    elseif Sb13 < S <= Sb14
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (S / S0).^(n13)

    elseif Sb14 < S <= Sb15
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (Sb14 / S0).^(n13 .- n14) .* (S / S0).^(n14)

    elseif Sb15 < S <= Sb16
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (Sb14 / S0).^(n13 .- n14) .* (Sb15 / S0).^(n14 .- n15) .* (S / S0).^(n15)

    elseif Sb16 < S <= Sb17
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (Sb14 / S0).^(n13 .- n14) .* (Sb15 / S0).^(n14 .- n15) .* (Sb16 / S0).^(n15 .- n16) .* (S / S0).^(n16)

    elseif Sb17 < S <= Sb18
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (Sb14 / S0).^(n13 .- n14) .* (Sb15 / S0).^(n14 .- n15) .* (Sb16 / S0).^(n15 .- n16).*(Sb17 / S0).^(n16 .- n17) .* (S / S0).^(n17)

    elseif Sb18 < S <= Sb19
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (Sb14 / S0).^(n13 .- n14) .* (Sb15 / S0).^(n14 .- n15) .* (Sb16 / S0).^(n15 .- n16).*(Sb17 / S0).^(n16 .- n17) .* (Sb18 / S0).^(n17 .- n18) .* (S / S0).^(n18)

    elseif Sb19 < S <= Sb20
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (Sb14 / S0).^(n13 .- n14) .* (Sb15 / S0).^(n14 .- n15) .* (Sb16 / S0).^(n15 .- n16).*(Sb17 / S0).^(n16 .- n17) .* (Sb18 / S0).^(n17 .- n18) .* (Sb19 / S0).^(n18 .- n19) .* (S / S0).^(n19)

    elseif Sb20 < S <= Sb21
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (Sb14 / S0).^(n13 .- n14) .* (Sb15 / S0).^(n14 .- n15) .* (Sb16 / S0).^(n15 .- n16).*(Sb17 / S0).^(n16 .- n17) .* (Sb18 / S0).^(n17 .- n18) .* (Sb19 / S0).^(n18 .- n19) .* (Sb20 / S0).^(n19 .- n20) .* (S / S0).^(n20)

    elseif Sb21 < S <= Sb22
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (Sb14 / S0).^(n13 .- n14) .* (Sb15 / S0).^(n14 .- n15) .* (Sb16 / S0).^(n15 .- n16).*(Sb17 / S0).^(n16 .- n17) .* (Sb18 / S0).^(n17 .- n18) .* (Sb19 / S0).^(n18 .- n19) .* (Sb20 / S0).^(n19 .- n20) .* (Sb21 / S0).^(n20 .- n21) .* (S / S0).^(n21)

    elseif Sb22 < S <= Sb23
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (Sb14 / S0).^(n13.- n14) .* (Sb15 / S0).^(n14 .- n15) .* (Sb16 / S0).^(n15 .- n16).*(Sb17 / S0).^(n16 .- n17) .* (Sb18 / S0).^(n17 .- n18) .* (Sb19 / S0).^(n18 .- n19) .* (Sb20 / S0).^(n19 .- n20) .*(Sb21 / S0).^(n20 .- n21) .* (Sb22 / S0).^(n21 .- n22) .* (S / S0).^(n22)

    elseif Sb23 < S <= Sb24
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (Sb14 / S0).^(n13 .- n14) .* (Sb15 / S0).^(n14 .- n15) .* (Sb16 / S0).^(n15 .- n16).*(Sb17 / S0).^(n16 .- n17) .* (Sb18 / S0).^(n17 .- n18) .* (Sb19 / S0).^(n18 .- n19) .* (Sb20 / S0).^(n19 .- n20) .*(Sb21 / S0).^(n20 .- n21) .* (Sb22 / S0).^(n21 .- n22) .* (Sb23 / S0).^(n22 .- n23) .* (S / S0).^(n23)

    elseif Sb24 < S <= Sb25
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (Sb14 / S0).^(n13 .- n14) .* (Sb15 / S0).^(n14 .- n15) .* (Sb16 / S0).^(n15 .- n16).*(Sb17 / S0).^(n16 .- n17) .* (Sb18 / S0).^(n17 .- n18) .* (Sb19 / S0).^(n18.- n19) .* (Sb20 / S0).^(n19 .- n20) .*(Sb21 / S0).^(n20 .- n21) .* (Sb22 / S0).^(n21 .- n22) .* (Sb23 / S0).^(n22 .- n23) .* (Sb24 / S0).^(n23 .- n24) .* (S / S0).^(n24)
 
    else S > Sb25
        result = As .* (Sb13 / S0).^(n12 .- n13) .* (Sb14 / S0).^(n13 .- n14) .* (Sb15 / S0).^(n14 .- n15) .* (Sb16 / S0).^(n15 .- n16).*(Sb17 / S0).^(n16 .- n17) .* (Sb18 / S0).^(n17 .- n18) .* (Sb19 / S0).^(n18 .- n19) .* (Sb20 / S0).^(n19 .- n20).*(Sb21 / S0).^(n20 .- n21) .* (Sb22 / S0).^(n21 .- n22) .* (Sb23 / S0).^(n22 .- n23) .* (Sb24 / S0).^(n23 .- n24).*(Sb25 / S0).^(n24 .- n25) .* (S / S0).^(n25)

    end
    return result
end


 function model_train_data(z1, f, idx, f_filtered,logbins, Sb1, Sb2,   Sb3,   Sb4,   Sb5,   Sb6,   Sb7,   Sb8,   Sb9,   Sb10,   Sb11,   Sb12,   Sb13,   Sb14,   Sb15,   Sb16,   Sb17,   Sb18,   Sb19,   Sb20,   Sb21,   Sb22,   Sb23,   Sb24,   Sb25)
    """
    Model function that generates data based on given parameters.

    Parameters:
    - z1: Input singal parameters. The first 26 correspond to 26 slopes, the 26th on the amplitude and the 28th is noise parameter A.
    - f: Frequency vector.
    - idx: Index vector that shows in which one out of the 1000 bins the frequency belonges to.
    - f_filtered: Filtered frequency vector.
    - logbins: Edges of the 1000 bins .
    - Sb1 to Sb25: Threshold values where the powerlaw changes.

    Returns:
    - Data_total: Combined data vector.
    - P: Randomly generated value sampled from a gaussian distribution (it is the noise parameter P).
    """

    z  =-12 .+z1[1:27]*(12-(-12))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    Amp =-14+z1[27]*(-6-(-14)) 
    A = z1[28]

    # Generate random value for the noise parameter P and save the slopes, the amplitude and the parameter in different vectors
    P  = 15+randn()*15*.2

    # Signal part. Following Data generation described in https://arxiv.org/pdf/2009.11845.pdf
    stds_omega = [sqrt.(f26(fi, Sb13,z[1:26]..., 10^(Amp), Sb1, Sb2, Sb3, Sb4, Sb5, Sb6, Sb7, Sb8, Sb9, Sb10, Sb11, Sb12, Sb13, Sb14, Sb15, Sb16, Sb17, Sb18, Sb19, Sb20, Sb21, Sb22, Sb23, Sb24, Sb25)) for fi in f]
    stds_omega_cuda = CuArray(Float32.(stds_omega))
    CUDA.device_reset!
    # Repeat for each chunk.
    std_eps = CUDA.randn(Float32, length(stds_omega), 94)
    CUDA.@sync samples_noise5 = stds_omega_cuda .* std_eps

    std_eps = CUDA.randn(Float32, length(stds_omega), 94)
    CUDA.@sync samples_noise6 = stds_omega_cuda .* std_eps

    CUDA.@sync c2 = (samples_noise5 .^ 2 + samples_noise6 .^ 2) ./ 2

    # Noise part. Following Data generation described in https://arxiv.org/pdf/2009.11845.pdf

    stds_n = sqrt.(Omega_noiseh2_AA.(f, 2.5 * 1e9, 3 * 1e8, P, A))  #compute sqrt(h^2Ωnoise(fi))
    stds_n_cuda = CuArray(Float32.(stds_n))
  
    std_eps = CUDA.randn(Float32, length(stds_n), 94)
    CUDA.@sync samples_noise3 = stds_n_cuda .* std_eps

    std_eps = CUDA.randn(Float32, length(stds_n), 94)
    CUDA.@sync samples_noise4 = stds_n_cuda .* std_eps

    CUDA.@sync c1 = (samples_noise3 .^ 2 + samples_noise4 .^ 2) / 2

    # Add signal and noise 
    CUDA.@sync c = c1 + c2


    # Calculate mean over chunks
    Data = Array(view(mean(c, dims=2), :, 1))  

    # Bin the data and frequencies after the 2971 element in 1000 bins and calculate a value for each of them using nverse variance weighting. (see https://arxiv.org/pdf/2009.11845.pdf)
    weight_norm = zeros(length(logbins))
    @threads for i in eachindex(logbins)
        res = sum((Omega_noiseh2_AA.(f_filtered[idx.==i], 2.5 * 1e9, 3 * 1e8, P, A)) .^ (-1))

        weight_norm[i] = res
    end


    weighted_data = zeros(length(logbins))
    @threads for i in eachindex(logbins)
        data_filtered = Data[2971:end][idx.==i]
        num = (Omega_noiseh2_AA.(f_filtered[idx.==i], 2.5 * 1e9, 3 * 1e8, P, A)) .^ (-1)
        w = num ./ weight_norm[i]

        weighted_data[i] = sum(data_filtered .* w)
    end

    # Concatenate  unbinned data before the 2971 element with the binned data. 
    Data_total = vcat(Data[1:2970], weighted_data)

    # Concatenate data with the value of the noise parameter P 
    return  vcat(log10.(Data_total), P) 

end


function model_plot(z1, f, idx, f_filtered,logbins, Sb1, Sb2,   Sb3,   Sb4,   Sb5,   Sb6,   Sb7,   Sb8,   Sb9,   Sb10,   Sb11,   Sb12,   Sb13,   Sb14,   Sb15,   Sb16,   Sb17,   Sb18,   Sb19,   Sb20,   Sb21,   Sb22,   Sb23,   Sb24,   Sb25)
    """
    Model function that generates data based on given parameters.

    Parameters:
    - z1: Input singal parameters. The first 26 correspond to 26 slopes, the 26th on the amplitude and the 28th is noise parameter A.
    - f: Frequency vector.
    - idx: Index vector that shows in which one out of the 1000 bins the frequency belonges to.
    - f_filtered: Filtered frequency vector.
    - logbins: Edges of the 1000 bins .
    - Sb1 to Sb25: Threshold values where the powerlaw changes.

    Returns:
    - Data_total: Combined data vector.
    - e: Randomly generated value sampled from a gaussian distribution (it is the noise parameter P).
    """
    z  =-12 .+z1[1:27]*(12-(-12))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    Amp =-14+z1[27]*(-6-(-14)) 
    A = z1[28]

    # Generate random value for the noise parameter P and save the slopes, the amplitude and the parameter in different vectors
    P  = 15+randn()*15*.2
    
    # Signal part. Following Data generation described in https://arxiv.org/pdf/2009.11845.pdf
    stds_omega = [sqrt.(f26(fi, Sb13,z[1:26]..., 10^(Amp), Sb1, Sb2, Sb3, Sb4, Sb5, Sb6, Sb7, Sb8, Sb9, Sb10, Sb11, Sb12, Sb13, Sb14, Sb15, Sb16, Sb17, Sb18, Sb19, Sb20, Sb21, Sb22, Sb23, Sb24, Sb25)) for fi in f]
    stds_omega_cuda = CuArray(Float32.(stds_omega))
    
    CUDA.device_reset!
    # Repeat for each chunk.
    std_eps = CUDA.randn(Float32, length(stds_omega), 94)
    CUDA.@sync samples_noise5 = stds_omega_cuda .* std_eps

    std_eps = CUDA.randn(Float32, length(stds_omega), 94)
    CUDA.@sync samples_noise6 = stds_omega_cuda .* std_eps

    CUDA.@sync c2 = (samples_noise5 .^ 2 + samples_noise6 .^ 2) ./ 2

    # Noise part. Following Data generation described in https://arxiv.org/pdf/2009.11845.pdf

    stds_n = sqrt.(Omega_noiseh2_AA.(f, 2.5 * 1e9, 3 * 1e8, P, A))  #compute sqrt(h^2Ωnoise(fi))
    stds_n_cuda = CuArray(Float32.(stds_n))

    std_eps = CUDA.randn(Float32, length(stds_n), 94)
    CUDA.@sync samples_noise3 = stds_n_cuda .* std_eps

    std_eps = CUDA.randn(Float32, length(stds_n), 94)
    CUDA.@sync samples_noise4 = stds_n_cuda .* std_eps

    CUDA.@sync c1 = (samples_noise3 .^ 2 + samples_noise4 .^ 2) / 2

    # Add signal and noise 
    CUDA.@sync c = c1 + c2

    # Calculate mean over chunks
    Data = Array(view(mean(c, dims=2), :, 1))  

    # Bin the data and frequencies after the 2971 element in 1000 bins and calculate a value for each of them using nverse variance weighting. (see https://arxiv.org/pdf/2009.11845.pdf)
    weight_norm = zeros(length(logbins))
    @threads for i in eachindex(logbins)
        res = sum((Omega_noiseh2_AA.(f_filtered[idx.==i], 2.5 * 1e9, 3 * 1e8, P, A)) .^ (-1))

        weight_norm[i] = res
    end

    weighted_data = zeros(length(logbins))
    @threads for i in eachindex(logbins)
        data_filtered = Data[2971:end][idx.==i]
        num = (Omega_noiseh2_AA.(f_filtered[idx.==i], 2.5 * 1e9, 3 * 1e8, P, A)) .^ (-1)
        w = num ./ weight_norm[i]

        weighted_data[i] = sum(data_filtered .* w)
    end


    weighted_f = zeros(length(logbins))
    @threads for i in eachindex(logbins)
        num = (Omega_noiseh2_AA.(f_filtered[idx.==i], 2.5 * 1e9, 3 * 1e8, P, A)) .^ (-1)
        w = num ./ weight_norm[i]

    weighted_f[i] = sum(f_filtered[idx.==i] .* w)
    end


    # Concatenate  unbinned data before the 2971 element with the binned data. 
    Data_total = vcat(Data[1:2970], weighted_data)

    f_total = vcat(f[1:2970], weighted_f)  

    # Concatenate data with the value of the noise parameter P 
    return  vcat(log10.(Data_total), P) , log10.(f_total) 

end
