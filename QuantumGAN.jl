using LinearAlgebra

using Yao
using Yao.Blocks
using QuAlgorithmZoo

"""
Quantum GAN.

Reference:
    Benedetti, M., Grant, E., Wossnig, L., & Severini, S. (2018). 
    Adversarial quantum circuit learning for pure state approximation, 1–14.
"""
struct QuGAN{N}
    target::DefaultRegister
    generator::MatrixBlock{N}
    discriminator::MatrixBlock
    reg0::DefaultRegister
    witness_op::MatrixBlock
    circuit::AbstractBlock
    gdiffs
    ddiffs

    function QuGAN(target::DefaultRegister, gen::MatrixBlock, dis::MatrixBlock)
        N = nqubits(target)
        c = sequence(gen, addbit(1), dis)
        witness_op = put(N+1, (N+1)=>P0)
        gdiffs = collect(gen, AbstractDiff)
        ddiffs = collect(dis, AbstractDiff)
        new{N}(target, gen, dis, zero_state(N), witness_op, c, gdiffs, ddiffs)
    end
end

"""loss function"""
#loss(qcg::QuGAN) = p0t(qcg) - p0g(qcg)
"""probability to get evidense qubit 0 on generation set."""
#p0g(qg::QuGAN) = expect(qg.witness_op, psi_discgen(qg)) |> real
"""probability to get evidense qubit 0 on target set."""
#p0t(qg::QuGAN) = expect(qg.witness_op, psi_disctarget(qg)) |> real
"""generated wave function"""
psi(qg::QuGAN) = copy(qg.reg0) |> qg.generator
"""input |> generator |> discriminator"""
psi_discgen(qg::QuGAN) = copy(qg.reg0) |> qg.circuit
"""target |> discriminator"""
psi_disctarget(qg::QuGAN) = copy(qg.target) |> qg.circuit[2:end]
"""tracedistance between target and generated wave function"""
distance(qg::QuGAN) = tracedist(qg.target, psi(qg))[]

"""obtain the gradient"""
function grad(qcg::QuGAN)
    ggrad_g = opdiff.(()->psi_discgen(qcg), qcg.gdiffs, Ref(qcg.witness_op))
    dgrad_g = opdiff.(()->psi_discgen(qcg), qcg.ddiffs, Ref(qcg.witness_op))
    dgrad_t = opdiff.(()->psi_disctarget(qcg), qcg.ddiffs, Ref(qcg.witness_op))
    [-ggrad_g; dgrad_t - dgrad_g]
end

"""the training process"""
function train(qcg::QuGAN{N}, g_learning_rate::Real, 
                             d_learning_rate::Real, niter::Int) where N
    ng = length(qcg.gdiffs)
    for i in 1:niter
        g = grad(qcg)
        dispatch!(+, qcg.generator, -g[1:ng]*g_learning_rate)
        dispatch!(-, qcg.discriminator, -g[ng+1:end]*d_learning_rate)
        (i*20)%niter==0 && println("Step = $i, Trance Distance = $(distance(qcg))")
    end
end

nbit = 3
target = rand_state(nbit)
gen = dispatch!(random_diff_circuit(nbit, 2, 
                       pair_ring(nbit)), :random) |> autodiff(:QC)
discriminator = dispatch!(random_diff_circuit(nbit+1, 
                        5, pair_ring(nbit+1)), :random) |> autodiff(:QC)
qcg = QuGAN(target, gen, discriminator)
train(qcg, 0.1, 0.2, 1000)
    