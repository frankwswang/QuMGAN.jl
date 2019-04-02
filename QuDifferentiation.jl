using Test, LinearAlgebra
using Yao
using QuAlgorithmZoo
using Yao.Blocks

witness_vec = zeros(1<<4)
witness_vec[[0, 3, 5, 10, 12, 15].+1] .= 1
witness_op = matrixgate(Diagonal(-witness_vec))

nbit = 4
depth = 4
cnots = [1=>2, 3=>4, 2=>3, 4=>1]
gen = random_diff_circuit(nbit, depth, cnots, mode=:Merged) |> autodiff(:QC)

dispatch!(gen, :random)
loss_func = () -> expect(witness_op, zero_state(nbit) |> gen)

function train(gen::AbstractBlock, g_learning_rate::Real, niter::Int)
    diff_blocks = collect(gen, AbstractDiff)
    for i in 1:niter
        ggrad = opdiff.(()->zero_state(nbit)|>gen, diff_blocks, Ref(witness_op))
        dispatch!(-, gen, ggrad.*g_learning_rate)
        if i%5==1 println("Step $i, loss = $(loss_func())") end
    end
end

train(gen, 0.1, 50)

using PyPlot
pygui(true)
pl = apply!(zero_state(nbit), gen)|>probs
bar(0:1<<nbit-1, pl)

gcf()
display(gcf())


