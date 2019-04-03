using LinearAlgebra
using Yao
using Yao.Blocks
using QuAlgorithmZoo
using Statistics

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

function expectCB(reg::DefaultRegister)
    dim = 1<<nqubits(reg)
    CpOp = diagm(0=>[(dim-1):-1:0;]) 
    expect(matrixgate(CpOp), reg) |>abs
end

function distance(qg::QuMGAN)
    regGenAll = psiGen(qg, qg.reg0)
    GenP1 = regGenAll[1]
    if (qg.N-qg.nBitGB) != (qg.RBit*(qg.Nblock-1))
        println("Dim mismatch!!")
    end
    GenP2 = product_state(ComplexF64, (qg.N-qg.nBitGB), (regGenAll[2] |> round |> Int64) )
    Gen = join(GenP2, GenP1)
    tracedist(qg.target, Gen)[]
end


# function expectCB(reg::DefaultRegister)
#     dim = 1<<nqubits(reg)
#     CpOp = diagm(0=>[dim:-1:0;]) 
#     expect(matrixgate(CpOp), reg) |>abs
# end
# function distance(qg::QuMGAN)
#     regGenAll = psiGen(qg, qg.reg0)
#     TarM = expectCB(qg.target)
#     GenM1 = expectCB(regGenAll[1])
#     GenM2 = regGenAll[2]
#     GenM = GenM1 + GenM2
#     [TarM, GenM, abs(TarM - GenM) ,(abs(TarM - GenM) / TarM)]
# end