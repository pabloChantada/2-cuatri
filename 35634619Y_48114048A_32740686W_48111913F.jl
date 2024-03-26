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
    inputs_train, targets_train = trainingDataset
    inputs_val, targets_val = validationDataset
    inputs_test, targets_test = testDataset

    println("Inputs Train: ", size(inputs_train))
    println("Targets Train: ", size(targets_train))
    println("Inputs Val: ", size(inputs_val))
    println("Targets Val: ", size(targets_val))
    println("Inputs Test: ", size(inputs_test))
    println("Targets Test: ", size(targets_test))

    # Transponemos los datos para poder usarlos con Flux
    inputs_train = transpose(inputs_train)
    targets_train = transpose(targets_train)
    inputs_val = transpose(inputs_val)
    targets_val = transpose(targets_val)
    inputs_test = transpose(inputs_test)
    targets_test = transpose(targets_test)
    
    
    # Creamos la RNA:
    ann = buildClassANN(size(inputs_train,1), topology, size(targets_train,1); transferFunctions=transferFunctions)

    # Creamos loss, función que calcula la pérdida de la red neuronal durante el entrenamiento. (la del enunciado)
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
    
    train_loss = loss(ann, inputs_train, targets_train)
    val_loss = loss(ann, inputs_val, targets_val)
    test_loss = loss(ann, inputs_test, targets_test)
    
    push!(train_losses, train_loss)
    push!(val_losses, val_loss)
    push!(test_losses, test_loss)
    epoch = 1
    while epoch < maxEpochs && val_loss > minLoss

        #Iniciamos el entrenamiento.
        # MIRAR AQUI
        Flux.train!(loss, ann, [(inputs_train, targets_train)], opt_state)
        
        # Calculamos los valores de pérdida de cada conjunto.
        # MIRAR AQUI
        train_loss = loss(ann, inputs_train, targets_train)
        val_loss = loss(ann, inputs_val, targets_val)
        test_loss = loss(ann, inputs_test, targets_test)
        
        # Llevamos los datos recogidos a su vector correspondiente mediante push.
        push!(train_losses, train_loss)
        push!(val_losses, val_loss)
        push!(test_losses, test_loss)
        if train_loss <= minLoss
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

    return trainClassANN(topology, (trainingDataset, validationDataset, testDataset),
        transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss,
        learningRate=learningRate, maxEpochsVal=maxEpochsVal)
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
    @assert size(outputs, 2) != 2 && size(outputs, 2) == size(targets, 2) "Las matrices deben tener el mismo número de columnas y no pueden tener una dimensión de 2 (caso binario)"
    if size(outputs, 2) == 1
        acc, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets)
        @assert (all([in(output, unique(targets)) for output in outputs])) "Las salidas no estan en las clases deseadas"
        return acc, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix
    else 
        sensitivity = zeros(size(outputs, 2))
        specificity = zeros(size(outputs, 2))
        positive_predictive_value = zeros(size(outputs, 2))
        negative_predictive_value = zeros(size(outputs, 2))
        f_score = zeros(size(outputs, 2))
        
        for i = 1:(size(outputs, 2))
            #matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix
            _, _, sensitivity[i], specificity[i], positive_predictive_value[i], negative_predictive_value[i], f_score[i], _ = confusionMatrix(outputs[:,i], targets[:,i])
        end       
        # esto esta mal
        # Define the unique classes

        # Create the confusion matrix
        matrix = zeros(Int, size(outputs, 2), size(outputs, 2))
        println(matrix)
        for real in size(outputs, 2)
            for predicted in (targets, 2)
                if real == predicted
                    matrix[real, predicted] = sum((outputs .== real) .& (targets .== predicted))
                else
                    matrix[real, predicted] = sum((outputs .== real) .& (targets .== predicted))
                end
            end
        end

        if weighted
            acc = accuracy(outputs, targets)
            fail_rate = 1 - acc
            weighted_sensitivity = sum(sensitivity .* size(outputs, 2)) / length(sensitivity)
            weighted_specificity = sum(specificity.* size(outputs, 2)) / length(specificity)
            weighted_positive_predictive_value = sum(positive_predictive_value.* size(outputs, 2)) / length(positive_predictive_value)
            weighted_negative_predictive_value = sum(negative_predictive_value.* size(outputs, 2)) / length(negative_predictive_value)
            weighted_f_score = sum(f_score.* size(outputs, 2)) / length(f_score)
            @assert (all([in(output, unique(targets)) for output in outputs])) "Error: Los valores de salida no están en el conjunto de valores posibles de salida."
            return (acc, fail_rate, weighted_sensitivity, weighted_specificity, weighted_positive_predictive_value, weighted_negative_predictive_value, weighted_f_score, matrix)
        else
            acc = accuracy(outputs, targets)
            fail_rate = 1 - acc
            macro_sensitivity = mean(sensitivity)
            macro_specificity = mean(specificity)
            macro_positive_predictive_value = mean(positive_predictive_value)
            macro_negative_predictive_value = mean(negative_predictive_value)
            macro_f_score = mean(f_score)
            @assert (all([in(output, unique(targets)) for output in outputs])) "Error: Los valores de salida no están en el conjunto de valores posibles de salida."
            return (acc, fail_rate, macro_sensitivity, macro_specificity, macro_positive_predictive_value, macro_negative_predictive_value, macro_f_score, matrix)
        end
        return matrix

    end
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    new_outputs = classifyOutputs(outputs)
    confusionMatrix(new_outputs, targets; weighted=weighted)
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    @assert length(unique(outputs)) == length(unique(targets)) "Las matrices deben tener las mismas dimensiones"

    classes_outputs = unique(outputs)
    classes_targets = unique(targets)

    new_outputs = oneHotEncoding(outputs, classes_outputs)
    new_targets = oneHotEncoding(targets, classes_targets)
    acc, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(new_outputs, new_targets; weighted=weighted)
    return (acc, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix)
    #
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
    matrix = confusionMatrix(outputs, targets; weighted=weighted)
    println("Matriz de confusión: \n")
    println(matrix)
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

# PARTE 11 - Todo mal
# --------------------------------------------------------------------------

function crossvalidation(N::Int64, k::Int64)
    # N -> numero de patrones
    # k -> numero de particiones
    @assert k >= 1 "Numero de particiones debe ser mayor o igual que 1"
    if N < k
        @warn "Número de patrones es menor que el número de particiones. Devolviendo particiones únicas."
        return collect(1:N)
    end
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

    index_vector = Vector{Any}(undef, size(targets, 1))
    for i = 1:(size(targets, 2))
        # Numero de elementos en cada particion
        elements = sum(targets[:, i] .== 1)
        col_positions = crossvalidation(elements, k)
        #=
        Aquí, findall(targets[:, i] .== 1) encuentra las posiciones donde el valor en la columna correspondiente de targets es 1, luego se asignan los valores de col_positions a esas posiciones en index_vector.
        =#
        index_vector[findall(targets[:, i] .== 1)] .= col_positions
    end
    @assert length(index_vector) == size(targets, 1) "Error en la particion"
    return index_vector
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    # Cualquier tipo de dato
    @assert k >= 1 "Numero de particiones debe ser mayor o igual que 1"
    unique_targets = unique(targets)
    if length(unique(targets)) >= 2
        index_vector = crossvalidation(oneHotEncoding(targets), k)
        return index_vector
    else
        index_vector = collect(1:length(targets))
        for i = 1:(length(unique_targets))
            elements = sum(targets .== unique_targets[i])
            col_positions = crossvalidation(elements, k)
            index_vector[findall(targets .== unique_targets[i])] .= col_positions
        end
        @assert length(index_vector) == length(targets) "Error en la particion"
        return index_vector
    end
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20)
    
    # Calcular el número de folds
    numFolds = maximum(crossValidationIndices)

    # Variables para almacenar las métricas
    precision = Float64[]
    errorRate = Float64[]
    sensitivity = Float64[]
    specificity = Float64[]
    VPP = Float64[]
    VPN = Float64[]
    F1 = Float64[]

    #Pasos únicos para crear una RNA
    # One-hot-encoding del vector de salidas deseadas
    targets_onehot = oneHotEncoding(targets)
    

    # Realizar la validación cruzada
    for fold in 1:numFolds
        
        # Separar los datos de entrenamiento y test
        train_indices = findall(i -> i != fold, crossValidationIndices)
        test_indices = findall(x -> x == fold, crossValidationIndices)

        train_inputs = inputs[:, train_indices]
        train_targets = targets_onehot[:, train_indices]

        test_inputs = inputs[:, test_indices]
        test_targets = targets_onehot[:, test_indices]


        # Bucle para repetir el entrenamiento dentro del fold
        for _ in 1:numExecutions
             
            # Entrenar la red neuronal
            
            if validationRatio > 0
                # Determinar el tamaño del conjunto de validación
                total_size = size(inputs, 2) + size(targets_onehot, 2)
                size_train = size(train_inputs, 2)


                validationRatio = (size_train * validationRatio) / total_size
                P = (1 - validationRatio)
                N = size_train

                # Obtener los índices para el conjunto de validación
                validation_indices = holdOut(N, P)
                
                # Conjunto de validación
                validation_inputs = train_inputs[:, validation_indices]
                validation_targets = train_targets[:, validation_indices]


                train_inputs, validation_inputs = holdOut
                ann_trained = trainClassANN(topology, (train_inputs, train_targets),
                    validationDataset=(validation_inputs, validation_targets), testDataset=(test_inputs, test_targets),
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss,
                    learningRate=learningRate,
                    maxEpochsVal=maxEpochsVal)
            else
                
                ann_trained = trainClassANN(topology, (train_inputs, train_targets),
                    validationDataset=(train_inputs, train_targets), testDataset=(test_inputs, test_targets),
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss,
                    learningRate=learningRate,
                    maxEpochsVal=maxEpochsVal)
            end 
            
            
            # Evaluar la red neuronal con el conjunto de prueba
            confusion_matrix = confusionMatrix(ann_trained[1](test_inputs), test_targets)

        
            # Coger las  métricas y almacenarlas en los vectores correspondientes
            acc = confusion_matrix[1]
            fail_rate = confusion_matrix[2]
            sensitivity_values = confusion_matrix[3]
            specificity_values = confusion_matrix[4]
            positive_predictive_value = confusion_matrix[5]
            negative_predictive_value = confusion_matrix[6]
            f_score = confusion_matrix[7]

            push!(precision, acc)
            push!(errorRate, fail_rate)
            push!(sensitivity, sensitivity_values)
            push!(specificity, specificity_values)
            push!(VPP, positive_predictive_value)
            push!(VPN, negative_predictive_value)
            push!(F1, f_score)
        end
    end

    # Calcular medias y desviaciones estándar de las métricas de todos los folds
    precision_mean = mean(precision)
    precision_std = std(precision)
    errorRate_mean = mean(errorRate)
    errorRate_std = std(errorRate)
    sensitivity_mean = mean(sensitivity)
    sensitivity_std = std(sensitivity)
    specificity_mean = mean(specificity)
    specificity_std = std(specificity)
    VPP_mean = mean(VPP)
    VPP_std = std(VPP)
    VPN_mean = mean(VPN)
    VPN_std = std(VPN)
    F1_mean = mean(F1)
    F1_std = std(F1)


    return ((precision_mean, precision_std), 
            (errorRate_mean, errorRate_std), 
            (sensitivity_mean, sensitivity_std), 
            (specificity_mean, specificity_std), 
            (VPP_mean, VPP_std), 
            (VPN_mean, VPN_std), 
            (F1_mean, F1_std))
end;




# PARTE 12
# --------------------------------------------------------------------------