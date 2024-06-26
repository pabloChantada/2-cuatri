# Numero Maximo de capas -> 2
# Usar funcion Dense para los Perceptrones Multicapa:
#   Numero de entradas
#   Salidas
#   Funcion de transferencia
# Transponer las matrices creadas
# Usar Float32 casi siempre, para las practicas usarlo siempre

using Flux;
using Flux.Losses;
using FileIO;
using DelimitedFiles;
using Statistics;
using Random;
using ScikitLearn;
using LinearAlgebra;
using ScikitLearn: fit!, predict;
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier 

# PARTE 1
# --------------------------------------------------------------------------

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    @assert numClasses > 1 "solo hay una clase"
    if numClasses == 2
        # Si solo hay dos clases, se devuelve una matriz con una columna.
        one_col_matrix = reshape(feature .== classes[1], :, 1)
        return one_col_matrix
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase.
        oneHot = Array{Bool,2}(undef, length(feature), numClasses)
        for numClass = 1:numClasses
            oneHot[:, numClass] .= (feature .== classes[numClass])
        end
        return oneHot
    end
end;

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    oneHotEncoding(feature, unique(feature))
end;

function oneHotEncoding(feature::AbstractArray{Bool,1})
    one_col_matrix = reshape(feature, :, 1)
    return one_col_matrix
end;

# PARTE 2
# --------------------------------------------------------------------------

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) # vectores de ints y floats
    # recibe una matriz
    # duvuelve una tupla con
    # matriz de 1 fila -> min de cada columna
    # matriz de 1 fila -> max de cada columna
    min_col = minimum(dataset, dims=1)
    max_col = maximum(dataset, dims=1)
    return (min_col, max_col)
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mean_col = mean(dataset, dims=1)
    std_col = std(dataset, dims=1)
    return (mean_col, std_col)
end;


# PARTE 3
# --------------------------------------------------------------------------

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    # Normalizar entre el max y el min
    # Matriz de valores a normalizar y parametros de normalizacion
    min_values, max_values = normalizationParameters[1], normalizationParameters[2]
    dataset .-= min_values
    range_values = max_values .- min_values
    # Caso de que los valores sean 0
    # range_values[range_values .== 0] .= 1
    dataset ./= (range_values)
    dataset[:, vec(min_values .== max_values)] .= 0
    return dataset
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax!(dataset, normalizationParameters)
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    # MEJOR COPY O DEEPCOPY EN ESTE CASO?
    new_dataset = copy(dataset)
    min_values, max_values = normalizationParameters[1], normalizationParameters[2]
    new_dataset .-= min_values
    range_values = max_values .- min_values
    # Caso de que los valores sean 0
    # range_values[range_values .== 0] .= 1
    new_dataset ./= (range_values)
    new_dataset[:, vec(min_values .== max_values)] .= 0
    return new_dataset
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax(dataset, normalizationParameters)
end;

# PARTE 4
# --------------------------------------------------------------------------

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    avg_values, std_values = normalizationParameters[1], normalizationParameters[2]
    dataset .-= avg_values
    dataset ./= std_values
    dataset[:, vec(std_values .== 0)] .= 0
    return dataset
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean!(dataset, normalizationParameters)
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    new_dataset = copy(dataset)
    avg_values, std_values = normalizationParameters[1], normalizationParameters[2]
    new_dataset .-= avg_values
    new_dataset ./= std_values
    new_dataset[:, vec(std_values .== 0)] .= 0
    return new_dataset
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean(dataset, normalizationParameters)
end;


# PARTE 5
# --------------------------------------------------------------------------

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    # outputs -> vector de salidas, no necesariamente un una RNA
    # threshold -> opcional
    return outputs .>= threshold
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        vector_outputs = classifyOutputs(outputs[:]; threshold)
        matrix_outputs = reshape(vector_outputs, :, 1)
        return matrix_outputs
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        matrix_outputs = falses(size(outputs))
        matrix_outputs[CartesianIndex.(indicesMaxEachInstance)] .= true
        return matrix_outputs
    end
end

# PARTE 6
# --------------------------------------------------------------------------

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #Las matrices targets y outputs deben tener la misma longitud.
    @assert length(targets) == length(outputs)
    #Divide el número de coincidencias entre el tamaño del vector targets para saber la media de aciertos.
    return sum(targets .== outputs) / length(targets)
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert size(targets) == size(outputs)
    if size(outputs, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1]);
    else
        mismatches = count(outputs .!= targets, dims = 2)
        total_samples = size(targets, 1)
        accuracy_values = accuracy_values = (count(mismatches .== 0)) / total_samples
        return accuracy_values
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #Los vectores targets y outputs deben tener la misma longitud.
    @assert length(targets) == length(outputs)
    new_outputs = classifyOutputs(outputs; threshold)
    return accuracy(new_outputs, targets)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    #Las matrices targets y outputs deben tener las mismas dimensiones
    @assert size(targets) == size(outputs)
    #Comprueba si la matriz outputs tiene una sola columna.
    if size(outputs, 2) == 1
        # outputs tiene una sola columna, llamamos a la función accuracy creada anteriormente.
        return accuracy(outputs[:, 1], targets[:, 1]; threshold=threshold)
    else
        outputs_bool = classifyOutputs(outputs; threshold)
        accuracy(outputs_bool, targets)
        return accuracy(outputs_bool, targets)
    end
end;

# PARTE 7
# --------------------------------------------------------------------------
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    # topology = [numero capas ocultas, numero de neuronas, (opcional) funciones de transferencia]
    # global ann, numInputsLayer
    @assert !isempty(topology) "No hay capas ocultas"

    ann = Chain()
    numInputsLayer = numInputs
    for i::Int = 1:length(topology)    
        numOutputsLayer = topology[i]    
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]))
        numInputsLayer = numOutputsLayer
    end
    # Ultima capa
    if numOutputs > 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity) );
        ann = Chain(ann..., softmax ); 
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs,transferFunctions[end]))
    end;
    return ann
end;

#PARTE 8 - 
# --------------------------------------------------------------------------

# El unico fallo que se me ocurre es en el tamaño de los vectores pero no deberia importar creo
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20)
     
    
    # Separamos los datos de los datasets
    inputs_train = trainingDataset[1]
    targets_train = trainingDataset[2]
    inputs_val = validationDataset[1]
    targets_val = validationDataset[2]
    inputs_test = testDataset[1]
    targets_test = testDataset[2]
    
    inputs_train2 = transpose(inputs_train)
    targets_train2 = transpose(targets_train)
    inputs_val2 = transpose(inputs_val)
    targets_val2 = transpose(targets_val)
    inputs_test2 = transpose(inputs_test)
    targets_test2 = transpose(targets_test)
    
    # a float si tal

    ann = buildClassANN(size(inputs_train2,1), topology, size(targets_train2,1); transferFunctions=transferFunctions)

    # Creamos loss, función que calcula la pérdida de la red neuronal durante el entrenamiento. (la del enunciado)
    # tiene que estar transpuesto
    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

    # Configuramos el estado del optimizador Adam (Adaptive Moment Estimation) 
    #Ponemos básicamente el ritmo ideal de aprendizaje de nuestra ann RNA.
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Creamos las variables para guardar el mejor modelo y su loss.
    best_model = deepcopy(ann)
    best_val_loss = Inf
    epochs_since_best = 0

    # Creamos los vectores que se utilizan para almacenar los valores de pérdida (loss) durante el 
    train_losses = Float64[]
    val_losses = Float64[]
    test_losses = Float64[]

    best_model = deepcopy(ann)

    train_loss = loss(ann, inputs_train2, targets_train2)
    push!(train_losses, train_loss)

    if validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0)
        val_loss = loss(ann, inputs_val2, targets_val2)
        push!(val_losses, val_loss)
    end

    if testDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0)
        test_loss = loss(ann, inputs_test2, targets_test2)
        push!(test_losses, test_loss)
    end
    
    # Volvemos a trasponerlas     

    for epoch in 1:maxEpochs

        #Iniciamos el entrenamiento.
        Flux.train!(loss, ann, [(inputs_train2, targets_train2)], opt_state)
        
        # Calculamos los valores de pérdida de cada conjunto.
        train_loss = loss(ann, inputs_train2, targets_train2)
        push!(train_losses, train_loss)

        if validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0)
            val_loss = loss(ann, inputs_val2, targets_val2)
            push!(val_losses, val_loss)
        end
        if testDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0)
            test_loss = loss(ann, inputs_test2, targets_test2)
            push!(test_losses, test_loss)
        end

        # Llevamos los datos recogidos a su vector correspondiente mediante push.
        if train_loss <= minLoss || 
            break
        end
        if val_loss <= minLoss && validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0)
            break
        end
        # Si el valor de pérdida de el modelo actual es menor que el de la variable 'mejor modelo' definida 
        #anteriormente cambiamos las variables antiguas por las de nuetro modelo actual.
        if val_loss < best_val_loss 
            best_val_loss = val_loss
            best_model = deepcopy(ann)
            epochs_since_best = 0           #inicializamos y reiniciamos a la vez un contador de hace cuantos modelos que surgió el mejor.
        
        #si el valor de pérdida no es mejor le sumamos una al contador, y 
        #si este contador sobrepasa o iguala el máximo permitido paramos.
        else
            epochs_since_best += 1
            if epochs_since_best >= maxEpochsVal
                break
            end
        end
        #Criterio de parada temprana: verificamos que el valor de pérdida actual no sea menor que el permitido.
        epoch += 1
    end
    return best_model, train_losses, val_losses, test_losses
end

# seguramente falle aqui
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20) 

    # Convertimos las salidas deseadas a vectores si es necesario
    # dataset = (inputs, reshape(dataset[2], (length(dataset[2]), 1)))
    # Convertimos las salidas deseadas a vectores si es necesario
    trainingDataset = (trainingDataset[1], reshape(trainingDataset[2], :, 1))
    validationDataset = (validationDataset[1], reshape(validationDataset[2], :, 1))
    testDataset = (testDataset[1], reshape(testDataset[2], :, 1))

    return trainClassANN(topology, trainingDataset, validationDataset=validationDataset,
        testDataset=testDataset, transferFunctions=transferFunctions,
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
        maxEpochsVal=maxEpochsVal)
end

# PARTE 9
# --------------------------------------------------------------------------

function holdOut(N::Int, P::Real)
    # N -> numero de patrones
    # P -> valor entre 0 y 1, indica el porcentaje para el test
    # numero de patrones para el test
    @assert P < 1 "Valores de test fuera de rango"
    test_size = Int(floor(N * P))
    # permutamos los datos
    index_perm = randperm(N)
    index_test = index_perm[1:test_size]
    index_train = index_perm[test_size+1:end]
    @assert length(index_test) + length(index_train) == N
    return (index_train, index_test)
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    # N -> numero de patrones
    # Pval -> valor entre 0 y 1, tasa de patrones del conjunto de validacion
    # Ptest -> valor entre 0 y 1, tasa de patrones del conjunto de prueba
    @assert Pval < 1 "Valores de validacion fuera de rango"
    @assert Ptest < 1 "Valores de test fuera de rango"
    # Permutacion aleatoria de los indices
    Nval = round(Int, N * Pval)
    Ntest = round(Int, N * Ptest)
    Ntrain = N - Nval - Ntest
    # Permutación aleatoria de los índices
    indexes = randperm(N)
    # Obtenemos los índices de los conjuntos
    index_train = indexes[1:Ntrain]
    index_val = indexes[Ntrain + 1:Ntrain + Nval]
    index_test = indexes[Ntrain + Nval + 1:end]
    # Comprobamos que los vectores resultantes sean igual a N
    @assert (length(index_train) + length(index_test) + length(index_val)) == N
    return (index_train, index_val, index_test)
end

# PARTE 10
# --------------------------------------------------------------------------
# 4.1 - Solucionado creo
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # Matrices cuadradadas = clases x clases
    # Muestra la distribucion de los patrones y la clasificacion que hace el modelo
    # Filas -> como ha efectuado el modelo
    # Columnas -> valores reales
    #=
    [VN FP;
     FN VP]
    =#
    matrix = [
    sum((outputs .== false) .& (targets .== false)) sum((outputs .== true) .& (targets .== false));
    sum((outputs .== false) .& (targets .== true)) sum((outputs .== true) .& (targets .== true))
    ]

    vn, fp, fn, vp = matrix[1,1], matrix[1,2], matrix[2,1], matrix[2,2]
    matrix_accuracy = (vn + vp) / (vn + vp + fn + fp)
    fail_rate = (fn + fp) / (vn + vp + fn + fp)
    
    sensitivity = vp / (fn + vp) |> x -> isnan(x) ? 1.0 : x
    specificity = vn / (vn + fp) |> x -> isnan(x) ? 1.0 : x
    positive_predictive_value = vp / (vp + fp) |> x -> isnan(x) ? 1.0 : x
    negative_predictive_value = vn / (vn + fn) |> x -> isnan(x) ? 1.0 : x
    if (sensitivity + positive_predictive_value) == 0
        f_score = 0
    else
        f_score = 2 * (positive_predictive_value * sensitivity) / (positive_predictive_value + sensitivity)
    end
    return matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix 
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    new_outputs = classifyOutputs(outputs; threshold)
    return confusionMatrix(new_outputs, targets)
    #
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1},
    targets::AbstractArray{Bool,1})
    #
    matrix = confusionMatrix(outputs, targets)[8]
    println("Matriz de confusión: \n")
    println("VN: ", matrix[1,1], " FP: ", matrix[1,2])
    println("FN: ", matrix[2,1], " VP: ", matrix[2,2])
    #
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    matrix = confusionMatrix(outputs, targets; threshold=threshold)[8]
    println("Matriz de confusión: \n")
    println("VN: ", matrix[1,1], " FP: ", matrix[1,2])
    println("FN: ", matrix[2,1], " VP: ", matrix[2,2])
    #
end;


# 4.2 - todo mal
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # outputs = matriz de salidas
    # targets = matriz de salidas deseadas
    @assert size(outputs, 2) == size(targets, 2) && size(outputs, 2) != 2 "Las matrices deben tener el mismo número de columnas y no pueden tener una dimensión de 2 (caso binario)"
    # Caso binario
    if size(outputs, 2) == 1
        @assert (all([in(output, unique(targets)) for output in outputs])) "Las salidas no estan en las clases deseadas"
        return confusionMatrix(vec(outputs), vec(targets))
    else 
        # Inicializar vectores de métricas
        num_classes = size(outputs, 2)
        sensitivities = zeros(Float64, num_classes)
        specificities = zeros(Float64, num_classes)
        ppvs = zeros(Float64, num_classes)
        npvs = zeros(Float64, num_classes)
        f1s = zeros(Float64, num_classes)
        
        for i in 1:num_classes
            # Call the confusionMatrix function with the columns corresponding to the current class
            _, _, sensitivities[i], specificities[i], ppvs[i], npvs[i], f1s[i], _ = confusionMatrix(outputs[:, i], targets[:, i])
        end
        
        # Este es el problema
        confusion_matrix = zeros(num_classes, num_classes)
        rows = size(targets, 1)
        for i in 1:rows
            x = findfirst(targets[i, :])
            y = findfirst(outputs[i, :])
            confusion_matrix[x,y] += 1
        end
        # f -> targets -> findfirst(targets[i,:])
        # c -> outputs -> findfirst(outputs[i,:])
        # confusion_matrix[f,c] += 1
        # Calcular precisión y tasa de fallo
        acc = accuracy(outputs, targets)
        fail_rate = 1 - acc
        # Calcular instacias de cada clase -> sum(targets, dims=1) = [clase1, clase2, ...]
        # vec(sum(targets, dims=1)) = [clase1, clase2, ...]
        # Sensivlidad1 * instancias1 + Sensibilidad2 * instancias2 + ... / total instancias
        if weighted
            # sum(matriz_confusion, dim=2) / sum(targets, 1)
            # Calcula el total de instancias por clase
            class_weights = sum(targets, dims=1)[:] / sum(targets)
            
            # Calcula métricas ponderadas
            # Posiblemente esto este mal
            # sin dividir
            sensitivities = sum(sensitivities .* class_weights) / sum(class_weights)
            specificities = sum(specificities .* class_weights) / sum(class_weights)
            ppvs = sum(ppvs .* class_weights) / sum(class_weights)
            npvs = sum(npvs .* class_weights) / sum(class_weights)
            f1s = sum(f1s .* class_weights) / sum(class_weights)
        else
            sensitivities = mean(sensitivities)
            specificities = mean(specificities)
            ppvs = mean(ppvs)
            npvs = mean(npvs)
            f1s = mean(f1s)
        end
        return acc, fail_rate, sensitivities, specificities, ppvs, npvs, f1s, confusion_matrix
    end
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    new_outputs = classifyOutputs(outputs)
    confusionMatrix(new_outputs, targets, weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert length(unique(outputs)) == length(unique(targets)) "Las matrices deben tener las mismas dimensiones"
    @assert all(in.(outputs, [unique(targets)])) "Todas las clases de 'outputs' deben estar en 'targets'"
    # da fallo en este assert el modelCrossValidation
    classes = unique([outputs; targets])  # Unión de clases únicas en ambos.

    new_outputs = oneHotEncoding(outputs, classes)
    new_targets = oneHotEncoding(targets, classes)
    return confusionMatrix(new_outputs, new_targets, weighted=weighted)
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
    matrix = confusionMatrix(outputs, targets; weighted=weighted)[8]
    num_classes = size(matrix, 1)
    for i in 1:num_classes
        for j in 1:num_classes
            print(confusion_matrix[i, j], " ")
        end
        println()
    end
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
    matrix = confusionMatrix(outputs, targets; weighted=weighted)
    println("Matriz de confusión: \n")
    println(matrix)
end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    matrix = confusionMatrix(outputs, targets; weighted=weighted)
    println("Matriz de confusión: \n")
    println(matrix)
end;

# PARTE 11
# --------------------------------------------------------------------------

function crossvalidation(N::Int64, k::Int64)
    # N -> numero de patrones
    # k -> numero de particiones
    @assert k >= 1 "Numero de particiones debe ser mayor o igual que 1"
    #=
    Necesaria ?
    if N < k
        @warn "Número de patrones es menor que el número de particiones. Devolviendo particiones únicas."
        return collect(1:N)
    end
    =#
    # Numero de elementos en cada particion
    k_vector = [1:k;]
    # Repetimos el vector k_vector hasta que sea mayor que N
    kn_vector = repeat(k_vector, ceil(Int, N / k))
    # Tomamos los primeros N elementos
    x = kn_vector[1:N]
    # Permutamos el vector
    shuffle!(x)
    @assert length(x) == N "Error en la particion"
    return x
    #
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    #
    @assert k >= 1 "Numero de particiones debe ser mayor o igual que 1"
    index_vector = collect(1:length(targets))
    # Creamos un vector con el numero de particion a la que pertenece cada patron
    positive = crossvalidation(count(targets .== true ), k)
    negative = crossvalidation(count(targets .== false ), k)
    # Asignamos a cada patron su particion
    index_vector[findall(targets .== true)] .= positive
    index_vector[findall(targets .== false)] .= negative
    @assert length(index_vector) == length(targets) "Error en la particion"
    return index_vector
    #
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    # N -> vecto de longitud = al numero de filas
    # Cada elemento indica el subconjunto al que pertenece

    @assert k >= 1 "Numero de particiones debe ser mayor o igual que 1"
    N = size(targets, 1)
    indices = Vector{Int64}(undef, N)

    for i in 1:(size(targets, 2))
        class_indices = crossvalidation(sum(targets[:, i]), k)
        indices[findall(targets[:, i])] = class_indices
    end
    return indices
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    @assert k >= 1 "Numero de particiones debe ser mayor o igual que 1"
    unique_classes = unique(targets)
    N = length(targets) 
    indices = Vector{Int64}(undef, N) 

    for class in unique_classes
        # True en las posicones donde la clase es igual a la clase actual
        # Se reinicia en cada iteración, la usamos para asignar los indices a cada clase
        class_mask = targets .== class
        # sum(class_mask) -> numero de patrones de la clase actual
        class_indices = crossvalidation(sum(class_mask), k)
        # Se asginan los valores de los indices en las instacias de class_mask
        # que son True
        indices[class_mask] = class_indices
    end
    return indices
end;

# Seguir desde aqui
function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20)
    
    # Convertir los targets a one-hot encoding
    targets_onehot = oneHotEncoding(targets)

    # Calcular el número de folds
    numFolds = maximum(crossValidationIndices)

    # Vectores para almacenar los resultados de las métricas para todos los folds
    precision, errorRate, sensitivity, specificity, ppv, npv, f1 = ([] for _ in 1:7)

    for fold in 1:numFolds
        # Separar datos de entrenamiento y test usando crossValidationIndices
        # crossvalidation .!= fold
        trainIdx = findall(x->x!=fold, crossValidationIndices)
        testIdx = findall(x->x==fold, crossValidationIndices)

        train_inputs = inputs[trainIdx, :]
        train_targets = targets_onehot[trainIdx, :]
        test_inputs = inputs[testIdx, :]
        test_targets = targets_onehot[testIdx, :]
        # Resultados de las métricas para este fold
        foldPrecision, foldErrorRate, foldSensitivity, foldSpecificity, foldPPV, foldNPV, foldF1 = ([] for _ in 1:7)

        for i in 1:numExecutions
            # División en entrenamiento y validación si es necesario
            if validationRatio > 0
                N = size(train_inputs, 1) # Número total de patrones en el conjunto de entrenamiento
                # adjustedValidationRatio = validationRatio * (N / (size(inputs, 1) - length(testIdx)))
                # P = 1 - adjustedValidationRatio
                # Dividir el conjunto de entrenamiento usando holdOut para obtener índices de validación
                (train_idx, val_idx) = holdOut(N, validationRatio)
                train_inputs_hold = train_inputs[train_idx, :]
                train_targets_hold = train_targets[train_idx, :]
                val_index = train_inputs[val_idx, :]
                val_tar = train_inputs[val_idx, :]
                # Preparar los datasets como tuplas
                trainingDataset = (train_inputs_hold, train_targets_hold)
                validationDataset = (val_index, val_tar)
                test_dataset = (test_inputs, test_targets)
                # Llamar a trainClassANN con conjuntos de entrenamiento y validación
                ann_trained, b... = trainClassANN(topology, trainingDataset, validationDataset=validationDataset,
                                testDataset=test_dataset,
                                transferFunctions=transferFunctions, maxEpochs=maxEpochs,
                                minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
            else
                # Preparar el dataset de entrenamiento como tupla
                trainingDataset = (train_inputs, train_targets)
                # Llamar a trainClassANN solo con el conjunto de entrenamiento
                ann_trained, b... = trainClassANN(topology, trainingDataset, transferFunctions=transferFunctions,
                                maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
                                maxEpochsVal=maxEpochsVal)
            end
            
            # Evaluación del modelo
            # La confusion matrix es lo que esta mal
            # [2] o [4]
            test_outputs = ann_trained(test_inputs')
            confusionMatrixResult = confusionMatrix(test_outputs', test_targets)
            # Assuming confusionMatrixResult is a tuple with the desired metrics
            push!(foldPrecision, confusionMatrixResult[1])
            push!(foldErrorRate, confusionMatrixResult[2])
            push!(foldSensitivity, confusionMatrixResult[3])
            push!(foldSpecificity, confusionMatrixResult[4])
            push!(foldPPV, confusionMatrixResult[5])
            push!(foldNPV, confusionMatrixResult[6])
            push!(foldF1, confusionMatrixResult[7])

            # Almacenar resultados
        end
        push!(precision, mean(foldPrecision))
        push!(errorRate, mean(foldErrorRate))
        push!(sensitivity, mean(foldSensitivity))
        push!(specificity, mean(foldSpecificity))
        push!(ppv, mean(foldPPV))
        push!(npv, mean(foldNPV))
        push!(f1, mean(foldF1))     
    end
    # Calcular medias y desviaciones estándar de las métricas de todos los folds
    result = ((mean(precision), std(precision)), (mean(errorRate), std(errorRate)),
              (mean(sensitivity), std(sensitivity)), (mean(specificity), std(specificity)),
              (mean(ppv), std(ppv)), (mean(npv), std(npv)), (mean(f1), std(f1)))
    return result
end;

# PARTE 12
# --------------------------------------------------------------------------

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
                              inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
                              crossValidationIndices::Array{Int64,1})


    if modelType == :ANN
        # Asegúrate de que 'topology' existe dentro de 'modelHyperparameters'
        if haskey(modelHyperparameters, "topology")
            topology = modelHyperparameters["topology"]
            # Llamada ajustada a ANNCrossValidation con el argumento esperado
            return ANNCrossValidation(topology, inputs, targets, crossValidationIndices;
                numExecutions = get(modelHyperparameters, "numExecutions", 50),
                transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(topology))),
                maxEpochs = get(modelHyperparameters, "maxEpochs", 1000),
                minLoss = get(modelHyperparameters, "minLoss", 0.0),
                learningRate = get(modelHyperparameters, "learningRate", 0.01),
                validationRatio = get(modelHyperparameters, "validationRatio", 0),
                maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", 20))
        else
            error("Hiperparámetro 'topology' no encontrado en 'modelHyperparameters'.")
        end
    end

    # Inicialización de vectores para resultados de métricas
    precision, errorRate, sensitivity, specificity, ppv, npv, f1 = ([] for _ in 1:7)

    # Conversión de targets a string para compatibilidad con Scikit-Learn
    targets = string.(targets)

    for fold in unique(crossValidationIndices)
        trainIndices = findall(x -> x != fold, crossValidationIndices)
        testIndices = findall(x -> x == fold, crossValidationIndices)

        X_train, X_test = inputs[trainIndices, :], inputs[testIndices, :]
        y_train, y_test = targets[trainIndices], targets[testIndices]

        # Creación y entrenamiento del modelo
        if modelType == :SVC
            model = SVC(C=modelHyperparameters["C"], kernel=modelHyperparameters["kernel"],
                        degree=modelHyperparameters["degree"], gamma=modelHyperparameters["gamma"],
                        coef0=modelHyperparameters["coef0"])
        elseif modelType == :DecisionTreeClassifier
            model = DecisionTreeClassifier(max_depth=modelHyperparameters["max_depth"], random_state=1)
        elseif modelType == :KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=modelHyperparameters["n_neighbors"])
        end
        # Entrenamiento del modelo con los conjuntos de entrenamiento
        fit!(model, X_train, y_train)

        # Realización de predicciones con el conjunto de prueba
        predictions = predict(model, X_test)
        # Calculo de métricas con las predicciones y ly_test))ts de prueba
        acc, fail_rate, sensitivities, specificities, ppvs, npvs, f1s = confusionMatrix(y_test, predictions)

        # Almacenamiento de resultados de métricas
        push!(precision, acc)
        push!(errorRate, fail_rate)
        push!(sensitivity, sensitivities)
        push!(specificity, specificities)
        push!(ppv, ppvs)
        push!(npv, npvs)
        push!(f1, f1s)     
    end

    # Cálculo de la media y desviación estándar de cada métrica
    result = ((mean(precision), std(precision)), (mean(errorRate), std(errorRate)),
              (mean(sensitivity), std(sensitivity)), (mean(specificity), std(specificity)),
              (mean(ppv), std(ppv)), (mean(npv), std(npv)), (mean(f1), std(f1)))
    return result
end