using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case 2: Training with non-empty datasets
    topology = [2, 3, 1]
    trainingDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    validationDataset = (rand(2, 50), rand(2, 50) .> 0.5)
    testDataset = (rand(2, 50), rand(2, 50) .> 0.5)
    transferFunctions = [σ, σ, σ]
    maxEpochs = 1000
    minLoss = 0.0
    learningRate = 0.01
    maxEpochsVal = 20

    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        validationDataset=validationDataset,
        testDataset=testDataset,
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        maxEpochsVal=maxEpochsVal
    )

    @test length(train_losses) > 1
    @test length(val_losses) > 1
    @test length(test_losses) > 1
    @test best_model isa Flux.Chain