using LinearAlgebra
using Yao
using Yao.Blocks
using QuAlgorithmZoo
using Statistics

"""QuMGAN Structure."""
struct QuMGAN{N}
    target::DefaultRegister
    generator::MatrixBlock{N}
    discriminator::MatrixBlock
    reg0::DefaultRegister
    witness_op::MatrixBlock
    circuit::AbstractBlock
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
    nCBlockGB
    nCBlockDB
    nCBlockG
    nCBlockD
    nM

    function QuMGAN(target::DefaultRegister, gen::MatrixBlock, dis::MatrixBlock, 
                   VBit::Int64, RBit::Int64, Nblock::Int64, 
                   nBitGB::Int64, nBitDB::Int64, depthGB::Int64, depthDB::Int64)
        N = nqubits(target)
        if Nblock == 1
            nCBlockGB = depthGB*2+1 
            nCBlockDB = depthDB*2+1
            nCBlockG = nCBlockGB * Nblock 
            nCBlockD = nCBlockDB * Nblock
        else
            nCBlockGB = 1
            nCBlockDB = 1
            nCBlockG = Nblock
            nCBlockD = Nblock
        end
        c = sequence(gen, addbit(1), dis)
        witness_op = put(nBitGB+1, (nBitGB+1)=>P0)
        gdiffs = collect(gen, AbstractDiff)
        ddiffs = collect(dis, AbstractDiff)
        nMeasure = 10000
        new{nBitGB}(target, gen, dis, zero_state(nBitGB), witness_op, c, N, gdiffs, ddiffs,
                    VBit, RBit, Nblock, nBitGB, nBitDB, depthGB, depthDB, 
                    nCBlockGB, nCBlockDB, nCBlockG, nCBlockD, nMeasure)
    end
end

"""Function to get MPS qubits and classical data from Generator."""
function psiGen(qg::QuMGAN, reg1::DefaultRegister)
    regBit = copy(reg1)
    #println("g1 regBit: $regBit")
    regBStore = 0.0
    blockN = 0
    if (qg.Nblock > 1)
        for i = 1:(qg.nCBlockG - qg.nCBlockG)
            regBit |>qg.generator[i] 
            #println("g2_1\n")
            if i%(2*depthGB+1)==0
                blockN += 1
                psRBit = (qg.nBitGB - qg.RBit + 1):qg.nBitGB
                #println("g2_2_1 psRBit: $psRBit\n")
                regBStore += (measure(focus!(regBit, psRBit); nshot = qg.nM)  |> mean)*2^(qg.RBit*(qg.Nblock-1-blockN))
                relax!(regBit, psRBit)
                #println("g2_2_2 regBit: $regBit\n")
                measure_reset!( regBit, psRBit, val = 0)
                #println("g2_3\n")
            end
            #println("g2_4\n")
        end
    end
    for i = (qg.nCBlockG - qg.nCBlockGB + 1):qg.nCBlockG
        regBit |>qg.generator[i]
    end
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
        for i = 1:(qg.nCBlockD - qg.nCBlockDB) 
            #println("regBit: $regBit")
            if i%(2*depthDB+1)==0
                regBit |> qg.discriminator[i] 
                #println("d2_1\n")
                blockN += 1
                psRBit = (qg.nBitDB - qg.RBit + 1):qg.nBitDB
                #println("d2_2_1\n")
                numStore = parse(Int, regBStore[(1+(blockN-1)*RBit):(blockN*RBit)], base = 2)
                #println("d2_2_2\n")
                measure_reset!( regBit, psRBit, val =  numStore)
                #println("d2_3\n")
            end
            #println("d2_4\n")
        end
    end
    for i = (qg.nCBlockD - qg.nCBlockDB + 1): qg.nCBlockD
        regBit |> qg.discriminator[i]
    end
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
        [TarM, GenM, abs(TarM - GenM) ,(abs(TarM - GenM) / TarM)]
        #abs(TarM - GenM) / TarM
    else
        # Tracedistance function when QuMGAN returns to QuGAN.
        tracedist(TarBitMPS(qg)[1], psiRegGen(qg))[]
    end
end

"""Function to obtain the gradient of the matrix blocks in QuMGAN circuit."""
function grad(qcg::QuMGAN)
    #println("gr1")
    ggrad_g = opdiff.(()->psiGenDis(qcg), qcg.gdiffs, Ref(qcg.witness_op))
    #println("gr1_2 regd: $(nqubits(psiGenDis(qcg)))")
    dgrad_g = opdiff.(()->psiGenDis(qcg), qcg.ddiffs, Ref(qcg.witness_op))
    #println("gr1_3 regt: $(nqubits(psiTarDis(qcg)))")
    dgrad_t = opdiff.(()->psiTarDis(qcg), qcg.ddiffs, Ref(qcg.witness_op))
    #println("gr2")
    [-ggrad_g; dgrad_t - dgrad_g]
end

"""Training Function."""
function train(qcg::QuMGAN{N}, g_learning_rate::Real, 
                             d_learning_rate::Real, niter::Int) where N
    ng = length(qcg.gdiffs)
    for i in 1:niter
        #println("t1")
        g = grad(qcg)
        #println("t2 g_l: $(length(g)) ng_l: $(length(length(qcg.gdiffs))) nd_l: $(length(length(qcg.ddiffs)))")
        dispatch!(+, qcg.generator, -g[1:ng]*g_learning_rate)
        dispatch!(-, qcg.discriminator, -g[ng+1:end]*d_learning_rate)
        #println("t3")
        (i*20)%niter==0 && println("Step = $i, Trace Distance = $(distance(qcg))")
    end
end

"""Basic prarameters setup."""
# Bit number of all.
nBit = 5
# Number of layers for GAN training in each generator block.
depthGB = 2
# Number of layers for GAN training in each discriminator block.
depthDB = 5
# Number V of bonding bits in each block.
VBit = 3    
# Number R of resusable bits in each block.
RBit = 1 

"""Secondary prarameters."""
# Number of MPS Bl0ocks / Times of bit reusing.
Nblock = Int((nBit - VBit - RBit) / RBit + 1) # Nblock::Int64. When nBit = VBit + RBit, circuit becomes non-MPS.
# Number of total bits in each generator block. 
nBitGB = VBit + RBit
# Number of total bits in each discriminator block / the total bit number for the circuit. 
nBitDB = nBitGB+1

"""Generate random Target wavefunction."""
target = rand_state(nBit)

"""Build Generator and Discriminator of QuMGAN circuit."""
genblock = random_diff_circuit(nBitGB, depthGB, pair_ring(nBitGB)) 
disblock = random_diff_circuit(nBitDB, depthDB, pair_ring(nBitDB)) 
gen = genblock
dis = disblock
if Nblock > 1
    for i in 1: (Nblock-1)
        global gen 
        global dis
        gen = chain(nBitGB, gen, genblock)
        dis = chain(nBitDB, dis, disblock)
    end
end
gen = dispatch!(gen, :random) |> autodiff(:QC)
dis = dispatch!(dis, :random) |> autodiff(:QC)

"""Declare QuMGAN."""
qcg = QuMGAN(target, gen, dis, VBit, RBit, Nblock, nBitGB, nBitDB, depthGB, depthDB)

"""Training."""
train(qcg, 0.1, 0.2, 1000)
