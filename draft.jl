#=
"""wavefunction through generator"""
function psiGen(qg::QuGAN)
    regGen = copy(qg.reg0)
    regGenStore = zero_state(0)
    for i=1:2qg.dG ##each layer consists of 1 layer of rotation gates and 1 layer of cnot gates.  
        regGen |> qg.generator[i] 
        if i % 2qg.dGB == 0
            regGenStore = join(regGenStore, copy(focus!(regGen, 1:qg.RBit)))
            relax!(regGen, 1:qg.RBit)
            measure_reset!(regGen, 1, val = 0)
        end
    end
    regGenStore = join(regGenStore, copy(focus!(regGen, (qg.RBit+1):qg.nBitGB))
    regGenStore 
end

"""Target qubit through discriminator"""
function psiTarDis(qg::QuGAN)
    regGenStore = psiGen(qg)
    regDis = join(copy(focus!(copy(regGenStore),(qg.RBit-qg.VBit-1+1):qg.nBitGB)),zero_state(1))
    regDisStore = zero_state(0)
    for i=1:(2qg.dD+1)
        regDis |> qg.discriminator[i] 
        if i % 2qg.dDB == 0
            regDisStore = join(regDisStore, copy(focus!(regDis, 1:qg.RBit)))
            RBitDis = focus!(copy(regGenStore),(qg.RBit-qg.VBit-1+1))
            measure_reset!(regBit, 1, val = measure!(RBitDis))
        end
    end
    regDisStore = join(regDisStore, copy(focus!(regDis, (qg.RBit+1):(qg.nBitGB+1)))
    regDisStore 
end
=#