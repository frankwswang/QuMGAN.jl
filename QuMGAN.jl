using LinearAlgebra
using Yao
using Yao.Blocks
using QuAlgorithmZoo
using Statistics

"""========== Program Inroduction =========="""
"""
Quantum Generative Adversarial Network based on Matrix Product States(GuMGAN).

Reference:
    https://zhuanlan.zhihu.com/p/50361547
    Benedetti, M., Grant, E., Wossnig, L., & Severini, S. (2018). 
    Adversarial quantum circuit learning for pure state approximation, 1–14.
"""


"""========== Sub-function for QuMGAN =========="""

"""Structure of parameters for QuMGAN."""
struct parSetup
    #= Input parameters =#
    nBit    # Bit number of all.
    depthGB # Depth for GAN training in each generator block.
    depthDB # Depth for GAN training in each discriminator block.
    VBit    # Number V of bonding bits in each block.
    RBit    # Number R of resusable bits in each block.

    #= Secondary parameters =#
    Nblock  # Number(Int64) of MPS Blocks / Times of bit reusing. When nBit = VBit + RBit, qubits returns to non-MPS.
    nBitGB  # Number of total bits in each generator block. 
    nBitDB  # Number of total bits in each discriminator block / the total bit number for the circuit. 
    function parSetup(nBit::Int64, depthGB::Int64, deothDB::Int64, VBit::Int64, RBit::Int64)
        typeof((nBit - VBit - RBit) / RBit) != Int64 && println("Error: Nblock is not Int!") 
        Nblock = Int((nBit - VBit - RBit) / RBit + 1)
        nBitGB = VBit + RBit
        nBitDB = nBitGB+1
        new(nBit, depthGB, deothDB, VBit, RBit, Nblock, nBitGB, nBitDB)
    end
end

"""Function to build Gnenrator and Discriminator for QuMGAN."""
function buildGD(nBitB::Int64, depthB::Int64, Nblock::Int64)
    CBlock = random_diff_circuit(nBitB, depthB, pair_ring(nBitB)) 
    CBlockV = []
    #println("b1 NBlock: $(Nblock)")
    for i = 1:Nblock
        CBlockV = push!(CBlockV, CBlock)
    end
    #println("b2  CBlockV: $(CBlockV)")
    CBlockAll = chain(nBitB,CBlockV)
    #println("b3")
    CBlockAll = dispatch!(CBlockAll, :random) |> autodiff(:QC)
    #println("b4")
end

"""QuMGAN Structure."""
struct QuMGAN{N}
    target::DefaultRegister
    generator::MatrixBlock{N}
    discriminator::MatrixBlock
    reg0::DefaultRegister
    witnessOp::MatrixBlock
    N
    gdiffs
    ddiffs
    VBit
    RBit
    Nblock
    nBitGB
    nBitDB
    depthGB
    depthDB
    nM

    function QuMGAN(target::DefaultRegister, VBit::Int64, RBit::Int64, 
                    depthGB::Int64, depthDB::Int64, Nblock::Int64, nBitGB::Int64, nBitDB::Int64)
        N = nqubits(target)
        gen = buildGD(nBitGB, depthGB, Nblock)
        dis = buildGD(nBitDB, depthDB, Nblock)
        witnessOp = put(nBitGB+1, (nBitGB+1)=>P0)
        gdiffs = collect(gen, AbstractDiff)
        ddiffs = collect(dis, AbstractDiff)
        nMeasure = 10000
        new{nBitGB}(target, gen, dis, zero_state(nBitGB), witnessOp, N, gdiffs, ddiffs,
                    VBit, RBit, Nblock, nBitGB, nBitDB, depthGB, depthDB, nMeasure)
    end
end

"""Function to get MPS qubits and classical data from Generator."""
function psiGen(qg::QuMGAN, reg1::DefaultRegister)
    regBit = copy(reg1)
    #println("g1 regBit: $regBit")
    regBStore = 0.0
    blockN = 0
    if (qg.Nblock > 1)
        for i = 1:(qg.Nblock - 1)
            regBit |>qg.generator[i] 
            #println("g2_1\n")
            #if i%(2*depthGB+1)==0
                blockN += 1
                psRBit = (qg.nBitGB - qg.RBit + 1):qg.nBitGB
                #println("g2_2_1 psRBit: $psRBit\n")
                regBStore += (measure(focus!(regBit, psRBit); nshot = qg.nM)  |> mean)*2^(qg.RBit*(qg.Nblock-1-blockN))
                relax!(regBit, psRBit)
                #println("g2_2_2 regBit: $regBit\n")
                measure_reset!( regBit, psRBit, val = 0)
                #println("g2_3\n")
            #end
            #println("g2_4\n")
        end
    end
    regBit |>qg.generator[qg.Nblock]
    #println("g3 regBStore: $regBStore")
    [regBit, regBStore]
end

"""Function to get MPS qubits from Discriminator."""
function psiDis(qg::QuMGAN, reg1::DefaultRegister, reg1store::Float64)
    regBit = join(copy(reg1), zero_state(1)) 
    #println("d1\n")
    regBStore = bitstring(reg1store |> round |> Int64)
    blockN = 0
    if (qg.Nblock > 1)
        for i = 1:(qg.Nblock - 1) 
            #println("regBit: $regBit")
            #if i%(2*depthDB+1)==0
                regBit |> qg.discriminator[i] 
                #println("d2_1\n")
                blockN += 1
                psRBit = (qg.nBitDB - qg.RBit + 1):qg.nBitDB
                #println("d2_2_1\n")
                numStore = parse(Int, regBStore[(1+(blockN-1)*RBit):(blockN*RBit)], base = 2)
                #println("d2_2_2\n")
                measure_reset!( regBit, psRBit, val =  numStore)
                #println("d2_3\n")
            #end
            #println("d2_4\n")
        end
    end
    regBit |> qg.discriminator[qg.Nblock]
    #println("d2_4\n")
    regBit 
end

"""Function to generate MPS qubits and classical data from Target wave function."""
function TarBitMPS(qg::QuMGAN)
    #println("T1 qg.target: $(qg.target)")
    target = copy(qg.target)
    #println("T2")
    TarStore = 0.0
    if qg.nBitGB < qg.N
        psRBit = (qg.nBitGB+1):qg.N
        #println("T3 psRBit: $(psRBit)\n   target: $(target)")
        TarStore= measure(focus!(copy(target), psRBit); nshot=qg.nM) |> mean
        #println("T4")
        measure_remove!(target, psRBit)
        #println("T5 target: $(nqubits(target)) qg.target: $(nqubits(qg.target))")
    end
    [target, TarStore]
end

"""Functions to get wavefunction in each situation."""
# reg0 |> Generator
psiRegGen(qg::QuMGAN) = psiGen(qg, qg.reg0)[1]
# reg0 |> Generator |> Discriminator
psiGenDis(qg::QuMGAN) = psiDis(qg, (psiGen(qg, qg.reg0))[1], (psiGen(qg, qg.reg0))[2])
# Target |> Discriminator
psiTarDis(qg::QuMGAN) = psiDis(qg, TarBitMPS(qg)[1], TarBitMPS(qg)[2])

"""Possible way to get(simulation cheating) full wave function of qubits from generator? [under development]"""
"""For Testing tracedistance"""
function GenBitAll(qh::QuMGAN)
end

"""Tracedistance between Target and Generator generated wave function."""
"""Need better method!!!"""
#=
"""Tracedistance between the whole wave function. Using collapsing data to reconstruct Gen wavefunction."""
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
=#

"""Difference between Target and Generator collapsing data."""
"""Need to figure out how to apply expectCB() to regGenAll[2] original wave function."""
# Expectation of wavefunction on conputational basis. 
function expectCB(reg::DefaultRegister)
    dim = 1<<nqubits(reg)
    CpOp = diagm(0=>[0:1:(dim-1);]) 
    expect(matrixgate(CpOp), reg) |>abs
end
# Difference between Target and Generator collapsing data.
function distance(qg::QuMGAN)
    if qg.Nblock > 1
        regGenAll = psiGen(qg, qg.reg0)
        TarM = expectCB(qg.target)
        GenM1 = expectCB(regGenAll[1])
        GenM2 = regGenAll[2]
        GenM = GenM1 + GenM2
        #[TarM, GenM, abs(TarM - GenM) ,(abs(TarM - GenM) / TarM)]
        abs(TarM - GenM) / TarM
    else
        # Tracedistance function when QuMGAN returns to QuGAN.
        tracedist(TarBitMPS(qg)[1], psiRegGen(qg))[]
    end
end

"""Function to obtain the gradient of the matrix blocks in QuMGAN circuit."""
function grad(qg::QuMGAN)
    #println("gr1")
    GgradG = opdiff.(()->psiGenDis(qg), qg.gdiffs, Ref(qg.witnessOp))
    #println("gr1_2 regd: $(nqubits(psiGenDis(qcg)))")
    DgradG = opdiff.(()->psiGenDis(qg), qg.ddiffs, Ref(qg.witnessOp))
    #println("gr1_3 regt: $(nqubits(psiTarDis(qcg)))")
    DgradT = opdiff.(()->psiTarDis(qg), qg.ddiffs, Ref(qg.witnessOp))
    #println("gr2")
    [-GgradG; DgradT - DgradG]
end

"""Training Function."""
function train(qg::QuMGAN{N}, 
               gLearningRate::Real, dLearningRate::Real, nIter::Int) where N
    ng = length(qg.gdiffs)
    println(("Step = 0, Trace Distance = $(distance(qg))"))
    for i = 1:nIter
        #println("t1")
        g = grad(qg)
        #println("t2 g_l: $(length(g)) ng_l: $(length(length(qg.gdiffs))) nd_l: $(length(length(qg.ddiffs)))")
        dispatch!(+, qg.generator, -g[1:ng]*gLearningRate)
        dispatch!(-, qg.discriminator, -g[ng+1:end]*dLearningRate)
        #println("t3")
        (i*20)%nIter==0 && println("Step = $i, Trace Distance = $(distance(qg))")
    end
end


"""========== Main Program =========="""

"""Setup parameters."""
par = parSetup(3, 2, 5, 2, 1)

"""Generate random Target wave function."""
target = rand_state(par.nBit)

"""Build QuMGAN Circuit."""
# qcg = QuMGAN(target, VBit, RBit, Nblock, nBitGB, nBitDB, depthGB, depthDB)
qcg = QuMGAN(target, par.VBit, par.RBit, par.depthGB, par.depthDB, 
             par.Nblock, par.nBitGB, par.nBitDB)

"""Training QuMGAN to copy target wave function."""
train(qcg, 0.1, 0.2, 1000)
