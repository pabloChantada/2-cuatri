using Test

# Assuming you have defined the trainClassANN function
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Step 4: Define tests for binary classification
@testset "Binary Classification Tests" begin
    # Step 1: Prepare your datasets
    # Example datasets (replace with your actual data)
    training_data = (rand(10, 5), rand(Bool, 2))
    validation_data = (rand(5, 5), rand(Bool, 5))
    test_data = (rand(5, 5), rand(Bool, 5))
    typeof(training_data)
    typeof(validation_data)
    typeof(test_data)
    # Step 2: Define network topology and other parameters
    topology = [3, 8, 4]  # Example: 5 input nodes, 3 hidden nodes, 2 output nodes
    max_epochs = 1000
    min_loss = 0.01
    learning_rate = 0.01
    max_epochs_val = 20

    # Test 1: Check if the output is within the range [0, 1]
    @test all(0 .<= trainClassANN(topology, training_data, validationDataset=validation_data,
                                  testDataset=test_data, transferFunctions=fill(σ, length(topology)),
                                  maxEpochs=max_epochs, minLoss=min_loss,
                                  learningRate=learning_rate, maxEpochsVal=max_epochs_val) .<= 1)
    
    # Test 2: Check if the output is binary (either 0 or 1)
    @test all(trainClassANN(topology, training_data, validationDataset=validation_data,
                           testDataset=test_data, transferFunctions=fill(σ, length(topology)),
                           maxEpochs=max_epochs, minLoss=min_loss,
                           learningRate=learning_rate, maxEpochsVal=max_epochs_val) .∈ [0, 1])
    
    # Test 3: Check if the output has the same shape as the input
    @test size(trainClassANN(topology, training_data, validationDataset=validation_data,
                            testDataset=test_data, transferFunctions=fill(σ, length(topology)),
                            maxEpochs=max_epochs, minLoss=min_loss,
                            learningRate=learning_rate, maxEpochsVal=max_epochs_val)) == size(training_data[1])
end
