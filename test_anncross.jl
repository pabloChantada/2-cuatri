function test_ANNCrossValidation()
    # Crear datos de prueba
    topology = [2, 2, 1]
    inputs = [0 0; 0 1; 1 0; 1 1]'
    targets = [1, 0, 0, 0, 1, 1, 0]
    crossValidationIndices = [1, 2, 1, 2]  # Solo para este ejemplo, debes proporcionar tus propios índices de validación cruzada

    # Ejecutar la función
    results = ANNCrossValidation(topology, inputs, targets, crossValidationIndices)

    # Verificar que la salida tenga el formato correcto
    @test length(results) == 7
    @test all(length(r) == 2 for r in results)

    # Verificar que las medias no sean NaN
    @test !isnan(results[1][1])
    @test !isnan(results[2][1])
    @test !isnan(results[3][1])
    @test !isnan(results[4][1])
    @test !isnan(results[5][1])
    @test !isnan(results[6][1])
    @test !isnan(results[7][1])

    # Verificar que las desviaciones estándar no sean NaN y sean no negativas
    @test !isnan(results[1][2])
    @test !isnan(results[2][2])
    @test !isnan(results[3][2])
    @test !isnan(results[4][2])
    @test !isnan(results[5][2])
    @test !isnan(results[6][2])
    @test !isnan(results[7][2])
    @test results[1][2] >= 0
    @test results[2][2] >= 0
    @test results[3][2] >= 0
    @test results[4][2] >= 0
    @test results[5][2] >= 0
    @test results[6][2] >= 0
    @test results[7][2] >= 0
end

# Ejecutar las pruebas
test_ANNCrossValidation()



include("35634619Y_48114048A_32740686W_48111913F.jl")

using Test

try
    test_ANNCrossValidation()
catch err
    show(err)
end