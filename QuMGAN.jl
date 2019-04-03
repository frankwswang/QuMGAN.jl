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
    depthG
    depthD
    nM

    function QuMGAN(target::DefaultRegister, gen::MatrixBlock, dis::MatrixBlock, 
                   VBit::Int64, RBit::Int64, Nblock::Int64, nBitGB::Int64, nBitDB::Int64, 
                   depthGB::Int64, depthDB::Int64, depthG::Int64, depthD::Int64)
        N = nqubits(target) 
        c = sequence(gen, addbit(1), dis)
        witness_op = put(nBitGB+1, (nBitGB+1)=>P0)
        gdiffs = collect(gen, AbstractDiff)
        ddiffs = collect(dis, AbstractDiff)
        nMeasure = 10000
        new{nBitGB}(target, gen, dis, zero_state(nBitGB), witness_op, c, N, gdiffs, ddiffs,
                    VBit, RBit, Nblock, nBitGB, nBitDB, depthGB, depthDB, depthG, depthD, nMeasure)
    end
end

"""Function to get MPS qubits and classical data from Generator."""
function psiGen(qg::QuMGAN, reg1::DefaultRegister)
    regBit = copy(reg1)
    #println("g1 regBit: $regBit")
    regBStore = 0.0
    blockN = 0
    for i = 1:(qg.Nblock-1)
        regBit |>qg.generator[i] 
        #println("g2_1\n")
        blockN += 1
        psRBit = (qg.nBitGB - qg.RBit + 1):qg.nBitGB
        #println("g2_2_1 psRBit: $psRBit\n")
        regBStore += (measure(focus!(regBit, psRBit); nshot = qg.nM)  |> mean)*2^(qg.RBit*(qg.Nblock-1-blockN))
        relax!(regBit, psRBit)
        #println("g2_2_2 regBit: $regBit\n")
        measure_reset!( regBit, psRBit, val = 0)
        #println("g2_3\n")
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
    for i = 1:(qg.Nblock-1) #(2qg.depthD + qg.Nblock)
        #println("regBit: $regBit")
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
    regBit |> qg.discriminator[qg.Nblock]
    #println("d2_4\n")
    regBit 
end

"""Function to generate MPS qubits and classical data from Target wave function."""
function TarBitMPS(qg::QuMGAN)
    #println("T1 qg.target: $(qg.target)")
    target = copy(qg.target)
    #println("T2")
    psRBit = (qg.nBitGB+1):qg.N
    #println("T3 psRBit: $(psRBit)\n   target: $(target)")
    TarStore= measure(focus!(copy(target), psRBit); nshot=qg.nM) |> mean
    #println("T4")
    measure_remove!(target, psRBit)
    #println("T5 target: $(nqubits(target)) qg.target: $(nqubits(qg.target))")
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
"""Tracedistance of MPS qubits part."""
distance(qg::QuMGAN) = tracedist(TarBitMPS(qg)[1], psiRegGen(qg))[]
=#

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
    CpOp = diagm(0=>[(dim-1):-1:0;]) 
    expect(matrixgate(CpOp), reg) |>abs
end
# Difference between Target and Generator collapsing data.
function distance(qg::QuMGAN)
    regGenAll = psiGen(qg, qg.reg0)
    TarM = expectCB(qg.target)
    GenM1 = expectCB(regGenAll[1])
    GenM2 = regGenAll[2]
    GenM = GenM1 + GenM2
    #[TarM, GenM, abs(TarM - GenM) ,(abs(TarM - GenM) / TarM)]
    abs(TarM - GenM) / TarM
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
        #println("t2")
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
Nblock = Int((nBit - VBit - RBit) / RBit + 1) # When nBit = VBit + RBit, the circuit return to non-MPS structure.
# Number of total bits in each generator block. 
nBitGB = VBit + RBit
# Number of total bits in each discriminator block / the total bit number for the circuit. 
nBitDB = nBitGB+1
# Number of total depth of Differentiable Circuit in Gnerator.
depthG = depthGB * Nblock
# Number of total depth of Differentiable Circuit in Gnerator, Discriminator part has another qubit for register. 
depthD = depthDB * Nblock

"""Generate random Target wavefunction."""
target = rand_state(nBit)

"""Build Generator and Discriminator of QuMGAN circuit."""
genblock = random_diff_circuit(nBitGB, depthGB, pair_ring(nBitGB)) |> autodiff(:QC)
disblock = random_diff_circuit(nBitDB, depthDB, pair_ring(nBitDB)) |> autodiff(:QC)
gen = chain(nBitGB)
dis = chain(nBitDB)
for i in 1: Nblock
    global gen 
    global dis
    gen = chain(gen, genblock)
    dis = chain(dis, disblock)
end

"""Declare QuMGAN."""
qcg = QuMGAN(target, gen, dis, VBit, RBit, Nblock, nBitGB, nBitDB, depthGB, depthDB, depthG, depthD)

"""Training."""
train(qcg, 0.1, 0.2, 2000)
