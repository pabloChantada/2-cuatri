
using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# 1º6  
@testset "Accuracy function" begin
    # Test case 1
    outputs = [true, true, false, false, true, false, false, true]
    targets = [true, false, true, false, true, false, true, false]
    @test isequal(0.5, accuracy(outputs, targets))

    # Test case 2
    outputs = [true, true, true, true]
    targets = [true, true, true, true]
    @test isequal(1.0, accuracy(outputs, targets))

    # Test case 3
    outputs = [false, false, false, false]
    targets = [true, true, true, true]
    @test isequal(0.0, accuracy(outputs, targets))

    # Test case 4
    outputs = [true, false, true, false]
    targets = [false, true, false, true]
    @test isequal(0.0, accuracy(outputs, targets))
end

# 2º
@testset "Accuracy function 1" begin
    # Test case 1
    outputs = [true false; 
                false false; 
                false false; 
                true true]
    targets = [true false; 
                true true; 
                true true; 
                true true]
    @test isequal(0.5, accuracy(outputs, targets))

    # Test case 2
    outputs = [true true; true true]
    targets = [true true; true true]
    @test isequal(1.0, accuracy(outputs, targets))

    # Test case 3
    outputs = [false; false; false; false]
    targets = [true; true; true; true]
    @test isequal(0.0, accuracy(outputs, targets))

    # Test case 4
    outputs = [true; false; true; false]
    targets = [false; true; false; true]
    @test isequal(0.0, accuracy(outputs, targets))
    
    # Test case 5
    outputs = [true true true true]
    targets = [false false false false]
    @test isequal(0.0, accuracy(outputs, targets))
    
    # Test case 6
    outputs = [true true true true]
    targets = [true true true true]
    @test isequal(1.0, accuracy(outputs, targets))
    
    # Test case 7
    outputs = [false false false false]
    targets = [false false false false]
    @test isequal(1.0, accuracy(outputs, targets))
    
    # Test case 8
    outputs = [false false false false]
    targets = [true true true true]
    @test isequal(0.0, accuracy(outputs, targets))
end

include("35634619Y_48114048A_32740686W_48111913F.jl")

# Acc es por los atributos -> 2/3 no  7/9
outputs = [false false false true; 
            false false false true; 
            false false false true;]

targets = [false false true false; 
            false false false true; 
            false false false true]

outputs = [false false false true; 
            false false true false; 
            false true false false;]

targets = [false false false true; 
            false false false true; 
            false false false true]

accuracy(outputs, targets)

# 3º
@testset "Accuracy function 2" begin
    # Test case 1
    outputs = [0.4, 0.6, 0.9, 0.2]
    targets = [true, false, true, false]
    @test isequal(0.5, accuracy(outputs, targets, threshold=0.5))

    # Test case 2
    outputs = [0.8, 0.6, 0.6, 0.5]
    targets = [true, true, true, true]
    @test isequal(1.0, accuracy(outputs, targets, threshold=0.5))

    # Test case 3
    outputs = [0.1, 0.2, 0.3, 0.4]
    targets = [true, true, true, true]
    @test isequal(0.0, accuracy(outputs, targets, threshold=0.5))

    # Test case 4
    outputs = [0.9, 0.7, 0.8, 0.6]
    targets = [false, true, false, true]
    @test isequal(0.0, accuracy(outputs, targets, threshold=0.8))
end

# 4º
@testset "Accuracy function 3" begin
    # Test 1: Matriz de 1x1 con valores correctos
    outputs1 = [0.6]
    targets1 = [true]
    @test accuracy(outputs1, targets1) == 1.0

    # Test 2: Matriz de 1x1 con valores incorrectos
    outputs2 = [0.2]
    targets2 = [true]
    @test accuracy(outputs2, targets2) == 0.0

    # Test 3: Matriz de 2x2 con valores correctos
    outputs3 = [0.8 0.4; 0.6 0.9]
    targets3 = [true false; 
                false true]
    @test accuracy(outputs3, targets3) == 1

    # Test 4: Matriz de 2x2 con valores incorrectos
    outputs4 = [0.6 0.2; 0.3 0.1]
    targets4 = [true false; 
                true true]
    @test accuracy(outputs4, targets4) == 0.5

    # Test 5: Matriz de 2x1 con valores correctos
    outputs5 = [0.8; 0.9]
    targets5 = [true; true]
    @test accuracy(outputs5, targets5) == 1.0

    # Test 6: Matriz de 2x1 con valores incorrectos
    outputs6 = [0.3; 0.1]
    targets6 = [true; true]
    @test accuracy(outputs6, targets6) == 0.0

    # Test 7: Matriz de 3x3 con todos los valores correctos
    outputs7 = [0.8 0.9 0.7; 0.6 0.5 0.4; 0.3 0.2 0.1]
    targets7 = [true false true; false true true; false true true]
    @test accuracy(outputs7, targets7) == 0.0

    # Test 8: Matriz de 3x3 con todos los valores incorrectos
    outputs8 = [0.2 0.1 0.3; 0.4 0.5 0.6; 0.9 0.8 0.7]
    targets8 = [false false true; false false true; true false false]
    @test accuracy(outputs8, targets8) == 1.0
end