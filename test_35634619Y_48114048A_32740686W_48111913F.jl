using Test

include("35634619Y_48114048A_32740686W_48111913F.jl")
targets = [1 0 0 0; 
            1 0 0 0;
            0 0 1 0; 
            0 0 1 0;
            1 0 0 0;
            0 0 0 1;
            0 1 0 0;
            1 0 0 0;]
k = 4

index_vector = Vector{Any}(undef, size(targets, 1))

for i = 1:(size(targets, 2))
    # Numero de elementos en cada particion
    elements = sum(targets[:, i] .== 1)
    col_positions = crossvalidation(elements, k)
    index_vector[findall(targets[:, i] .== 1)] .= col_positions
end
println(index_vector)


using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case 1: targets with 4 columns and 8 rows
targets1 = [true false true;
            false true false;
            true false true;
            true false true;
            true false true;
            false true false;
            true false true;
            true false true]
k1 = 4
println(crossvalidation(targets1, k1))

using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case 1: targets with 4 columns and 8 rows
targets1 = [true false true;
            false true false;
            true false true;
            true false true;
            true false true;
            false true false;
            true false true;
            true false true]
k1 = 4
crossvalidation(targets1, k1)

include("35634619Y_48114048A_32740686W_48111913F.jl")

using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case 1: Confusion matrix for all true outputs and targets
function test_confusionMatrix_case1()
    outputs = [true, true, true, true]
    targets = [true, true, true, true]
    expected_matrix_accuracy = 1.0
    expected_fail_rate = 0.0
    expected_sensitivity = 1.0
    expected_specificity = 1.0
    expected_positive_predictive_value = 1.0
    expected_negative_predictive_value = 1.0
    expected_f_score = 1.0
    expected_matrix = [0 0; 0 4]
    
    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets)
    
    @test matrix_accuracy ≈ expected_matrix_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivity ≈ expected_sensitivity
    @test specificity ≈ expected_specificity
    @test positive_predictive_value ≈ expected_positive_predictive_value
    @test negative_predictive_value ≈ expected_negative_predictive_value
    @test f_score ≈ expected_f_score
    @test matrix ≈ expected_matrix
end

# Test case 2: Confusion matrix for all false outputs and targets
function test_confusionMatrix_case2()
    outputs = [false, false, false, false]
    targets = [false, false, false, false]
    expected_matrix_accuracy = 1.0
    expected_fail_rate = 0.0
    expected_sensitivity = 1.0
    expected_specificity = 1.0
    expected_positive_predictive_value = 1.0
    expected_negative_predictive_value = 1.0
    expected_f_score = 1.0
    expected_matrix = [4 0; 0 0]
    
    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets)
    
    @test matrix_accuracy ≈ expected_matrix_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivity ≈ expected_sensitivity
    @test specificity ≈ expected_specificity
    @test positive_predictive_value ≈ expected_positive_predictive_value
    @test negative_predictive_value ≈ expected_negative_predictive_value
    @test f_score ≈ expected_f_score
    @test matrix ≈ expected_matrix
end

# Test case 3: Confusion matrix for mixed outputs and targets
function test_confusionMatrix_case3()
    outputs = [true, false, true, false]
    targets = [false, true, false, true]
    expected_matrix_accuracy = 0.0
    expected_fail_rate = 1.0
    expected_sensitivity = 0.0
    expected_specificity = 0.0
    expected_positive_predictive_value = 0.0
    expected_negative_predictive_value = 0.0
    expected_f_score = 0.0
    expected_matrix = [0 2; 2 0]
    
    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets)
    
    @test matrix_accuracy ≈ expected_matrix_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivity ≈ expected_sensitivity
    @test specificity ≈ expected_specificity
    @test positive_predictive_value ≈ expected_positive_predictive_value
    @test negative_predictive_value ≈ expected_negative_predictive_value
    @test f_score ≈ expected_f_score
    @test matrix ≈ expected_matrix
end

# Run the tests
@testset "Confusion Matrix Tests" begin
    test_confusionMatrix_case1()
    test_confusionMatrix_case2()
    test_confusionMatrix_case3()
end