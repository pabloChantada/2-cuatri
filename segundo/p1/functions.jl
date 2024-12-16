
# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses


# -------------------------------------------------------------------------
# Funciones para codificar entradas y salidas categóricas

# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    # Primero se comprueba que todos los elementos del vector esten en el vector de clases (linea adaptada del final del ejercicio 4)
    @assert(all([in(value, classes) for value in feature]));
    numClasses = length(classes);
    @assert(numClasses>1)
    if (numClasses==2)
        # Si solo hay dos clases, se devuelve una matriz con una columna
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase
        # Cualquiera de estos dos tipos (Array{Bool,2} o BitArray{2}) vale perfectamente
        # oneHot = Array{Bool,2}(undef, size(feature,1), numClasses);
        # oneHot =  BitArray{2}(undef, size(feature,1), numClasses);
        # for numClass = 1:numClasses
        #     oneHot[:,numClass] .= (feature.==classes[numClass]);
        # end;
        # Una forma de hacerlo sin bucles sería la siguiente:
        oneHot = convert(BitArray{2}, hcat([instance.==classes for instance in feature]...)');
    end;
    return oneHot;
end;
# Esta funcion es similar a la anterior, pero si no es especifican las clases, se toman de la propia variable
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));
# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
#  En este caso, el propio vector ya está codificado, simplemente lo convertimos a una matriz columna
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);
# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
#  la llamada a la función correspondiente


# -------------------------------------------------------------------------
# Funciones para calcular los parametros de normalizacion y normalizar

# Para calcular los parametros de normalizacion, segun la forma de normalizar que se desee:
calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) = ( minimum(dataset, dims=1), maximum(dataset, dims=1) );
calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2}) = ( mean(dataset, dims=1), std(dataset, dims=1) );

# 4 versiones de la funcion para normalizar entre 0 y 1:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;
normalizeMinMax!(dataset::AbstractArray{<:Real,2})                                                              = normalizeMinMax!(     dataset , calculateMinMaxNormalizationParameters(dataset));
normalizeMinMax( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) = normalizeMinMax!(copy(dataset), normalizationParameters)
normalizeMinMax( dataset::AbstractArray{<:Real,2})                                                              = normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));

# 4 versiones similares de la funcion para normalizar de media 0:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset;
end;
normalizeZeroMean!(dataset::AbstractArray{<:Real,2})                                                              = normalizeZeroMean!(     dataset , calculateZeroMeanNormalizationParameters(dataset));
normalizeZeroMean( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) = normalizeZeroMean!(copy(dataset), normalizationParameters)
normalizeZeroMean( dataset::AbstractArray{<:Real,2})                                                              = normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));


# -------------------------------------------------------
# Funcion que permite transformar una matriz de valores reales con las salidas del clasificador o clasificadores en una matriz de valores booleanos con la clase en la que sera clasificada

classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5) = outputs.>=threshold;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return reshape(classifyOutputs(outputs[:]; threshold=threshold), :, 1);
    else
        # Miramos donde esta el valor mayor de cada instancia con la funcion findmax
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Creamos la matriz de valores booleanos con valores inicialmente a false y asignamos esos indices a true
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Comprobamos que efectivamente cada patron solo este clasificado en una clase
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;


# -------------------------------------------------------
# Funciones para calcular la precision

accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs.==targets);
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = accuracy(outputs.>=threshold, targets);
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]; threshold=threshold);
    else
        return accuracy(classifyOutputs(outputs), targets);
    end;
end;


# -------------------------------------------------------
# Funciones para crear y entrenar una RNA

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:(length(topology))
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    (inputs, targets) = dataset;

    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide
    @assert(size(inputs,1)==size(targets,1));

    # Creamos la RNA
    ann = buildClassANN(size(inputs,2), topology, size(targets,2));

    # Definimos la funcion de loss
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float32[];

    # Empezamos en el ciclo 0
    numEpoch = 0;
    # Calculamos el loss para el ciclo 0 (sin entrenar nada)
    trainingLoss = loss(ann, inputs', targets');
    #  almacenamos el valor de loss y precision en este ciclo
    push!(trainingLosses, trainingLoss);
    #  y lo mostramos por pantalla
    println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    opt_state = Flux.setup(Adam(learningRate), ann);

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)

        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, ann, [(inputs', targets')], opt_state);

        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
        # Calculamos las metricas en este ciclo
        trainingLoss = loss(ann, inputs', targets');
        #  almacenamos el valor de loss
        push!(trainingLosses, trainingLoss);
        #  lo mostramos por pantalla
        println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    end;

    # Devolvemos la RNA entrenada y el vector con los valores de loss
    return (ann, trainingLosses);
end;


trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) = trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)); maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N);
    numTrainingInstances = Int(round(N*(1-P)));
    return (indices[1:numTrainingInstances], indices[numTrainingInstances+1:end]);
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert ((Pval>=0.) & (Pval<=1.));
    @assert ((Ptest>=0.) & (Ptest<=1.));
    @assert ((Pval+Ptest)<=1.);
    # Primero separamos en entrenamiento+validation y test
    (trainingValidationIndices, testIndices) = holdOut(N, Ptest);
    # Después separamos el conjunto de entrenamiento+validation
    (trainingIndices, validationIndices) = holdOut(length(trainingValidationIndices), Pval*N/length(trainingValidationIndices))
    return (trainingValidationIndices[trainingIndices], trainingValidationIndices[validationIndices], testIndices);
end;



# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y test. Estos dos ultimos son opcionales
# Es la funcion anterior, modificada para calcular errores en los conjuntos de validacion y test y realizar parada temprana si es necesario
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;

    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide tanto en entrenamiento como en validacion como test
    @assert(size(trainingInputs,   1)==size(trainingTargets,   1));
    @assert(size(testInputs,       1)==size(testTargets,       1));
    @assert(size(validationInputs, 1)==size(validationTargets, 1));
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y validación, si este no está vacío
    !isempty(validationInputs)  && @assert(size(trainingInputs, 2)==size(validationInputs, 2));
    !isempty(validationTargets) && @assert(size(trainingTargets,2)==size(validationTargets,2));
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y test, si este no está vacío
    !isempty(testInputs)  && @assert(size(trainingInputs, 2)==size(testInputs, 2));
    !isempty(testTargets) && @assert(size(trainingTargets,2)==size(testTargets,2));

    # Creamos la RNA
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2); transferFunctions=transferFunctions);

    # Definimos la funcion de loss
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses   = Float32[];
    validationLosses = Float32[];
    testLosses       = Float32[];

    # Empezamos en el ciclo 0
    numEpoch = 0;

    # Una funcion util para calcular los resultados y mostrarlos por pantalla si procede
    function calculateLossValues()
        # Calculamos el loss en entrenamiento, validacion y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        trainingLoss = loss(ann, trainingInputs', trainingTargets');
	validationLoss = NaN; testLoss = NaN;
        push!(trainingLosses, trainingLoss);
        !isempty(validationInputs) && (validationLoss = loss(ann, validationInputs', validationTargets'); push!(validationLosses, validationLoss);)
	!isempty(      testInputs) && (testLoss       = loss(ann,       testInputs',       testTargets'); push!(      testLosses,       testLoss);)
        return (trainingLoss, validationLoss, testLoss);
    end;

    # Calculamos los valores de loss para el ciclo 0 (sin entrenar nada)
    (trainingLoss, validationLoss, _) = calculateLossValues();

    if isempty(validationInputs) maxEpochsVal=Inf; end;

    # Numero de ciclos sin mejorar el error de validacion y el mejor error de validation encontrado hasta el momento
    numEpochsValidation = 0; bestValidationLoss = validationLoss;
    # Cual es la mejor ann que se ha conseguido
    bestANN = deepcopy(ann);

    opt_state = Flux.setup(Adam(learningRate), ann)

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (numEpochsValidation<maxEpochsVal)

        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt_state);

        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;

        # Calculamos los valores de loss para este ciclo
        (trainingLoss, validationLoss, _) = calculateLossValues();

        # Aplicamos la parada temprana si hay conjunto de validacion
        if !isempty(validationInputs)
            if validationLoss<bestValidationLoss
                bestValidationLoss = validationLoss;
                numEpochsValidation = 0;
                bestANN = deepcopy(ann);
            else
                numEpochsValidation += 1;
            end;
        end;

    end;

    # Si no hubo conjunto de validacion, la mejor RNA será siempre la del último ciclo
    if isempty(validationInputs)
        bestANN = ann;
    end;

    return (bestANN, trainingLosses, validationLosses, testLosses);
end;


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;

    return trainClassANN(topology, (trainingInputs, reshape(trainingTargets, length(trainingTargets), 1)); validationDataset=(validationInputs, reshape(validationTargets, length(validationTargets), 1)), testDataset=(testInputs, reshape(testTargets, length(testTargets), 1)), transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal);
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------



function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    numInstances = length(targets);
    @assert(length(outputs)==numInstances);
    # Valores de la matriz de confusion
    TN = sum(.!outputs .& .!targets); # Verdaderos negativos
    FN = sum(.!outputs .&   targets); # Falsos negativos
    TP = sum(  outputs .&   targets); # Verdaderos positivos
    FP = sum(  outputs .& .!targets); # Falsos negativos
    # Creamos la matriz de confusión, poniendo en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
    #  Primera fila/columna: negativos
    #  Segunda fila/columna: positivos
    confMatrix = [TN FP;
                  FN TP];
    # Metricas que se derivan de la matriz de confusion:
    acc         = (TN+TP)/(TN+FN+TP+FP);
    errorRate   = 1. - acc;
    # Para sensibilidad, especificidad, VPP y VPN controlamos que algunos casos pueden ser NaN
    #  Para el caso de sensibilidad y especificidad, en un conjunto de entrenamiento estos no pueden ser NaN, porque esto indicaria que se ha intentado entrenar con una unica clase
    #   Sin embargo, sí pueden ser NaN en el caso de aplicar un modelo en un conjunto de test, si este sólo tiene patrones de una clase
    #  Para VPP y VPN, sí pueden ser NaN en caso de que el clasificador lo haya clasificado todo como negativo o positivo respectivamente
    # En estos casos, estas metricas habria que dejarlas a NaN para indicar que no se han podido evaluar
    #  Sin embargo, como es posible que se quiera combinar estos valores al evaluar una clasificacion multiclase, es necesario asignarles un valor. El criterio que se usa aqui es que estos valores seran igual a 1
    #   Se utiliza este criterio porque, por ejemplo en el caso de recall (sensibilidad), no habría fallado en ningún positivo, porque no hay ninguno
    recall      = (TP==FN==0) ? 1. : TP/(TP+FN); # Sensibilidad
    specificity = (TN==FP==0) ? 1. : TN/(TN+FP); # Especificidad
    precision   = (TP==FP==0) ? 1. : TP/(TP+FP); # Valor predictivo positivo
    NPV         = (TN==FN==0) ? 1. : TN/(TN+FN); # Valor predictivo negativo
    # Calculamos F1
    F1          = (recall==precision==0) ? 0. : 2*(recall*precision)/(recall+precision);
    @assert(!isnan(F1));
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;


confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = confusionMatrix(outputs.>=threshold, targets);


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    (numInstances, numClasses) = size(targets);
    # Nos aseguramos de que no hay dos columnas
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    end;

    # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
    @assert(all(sum(outputs, dims=2).==1));
    # Reservamos memoria para las metricas de cada clase, inicializandolas a 0 porque algunas posiblemente no se calculen
    recall      = zeros(numClasses);
    specificity = zeros(numClasses);
    precision   = zeros(numClasses);
    NPV         = zeros(numClasses);
    F1          = zeros(numClasses);
    # Calculamos las metricas para cada clase, usando la función anterior para problemas de 2 clases
    for numClass in 1:numClasses
        # Calculamos las metricas de cada problema binario correspondiente a cada clase y las almacenamos en los vectores correspondientes
        (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
    end;
    # Es posible hacer esta operación sin bucle, de esta manera:
    # [(_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]); for numClass in 1:numClasses]

    # Creamos la matriz de confusión
    confMatrix = [sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]) for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses];

    # Aplicamos las formas de combinar las metricas macro o weighted
    if weighted
        # Calculamos los valores de ponderacion para hacer el promedio
        numInstancesFromEachClass = vec(sum(targets, dims=1));
        @assert(numInstances == sum(numInstancesFromEachClass));
        weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
        recall      = sum(weights.*recall);
        specificity = sum(weights.*specificity);
        precision   = sum(weights.*precision);
        NPV         = sum(weights.*NPV);
        F1          = sum(weights.*F1);
    else
        recall      = mean(recall);
        specificity = mean(specificity);
        precision   = mean(precision);
        NPV         = mean(NPV);
        F1          = mean(F1);
    end;
    # Precision y tasa de error las calculamos con las funciones definidas previamente
    acc = accuracy(outputs, targets);
    errorRate = 1 - acc;

    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;

confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)



function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
#    @assert(all([in(output, unique(targets)) for output in outputs]));
    classes = unique([targets; outputs]);
    # Es importante calcular el vector de clases primero y pasarlo como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;



# Funciones auxiliares para visualizar por pantalla la matriz de confusion y las metricas que se derivan de ella
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;
printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) =  printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)



printConfusionMatrix(outputs::AbstractArray{Bool,1},   targets::AbstractArray{Bool,1})                      = printConfusionMatrix(reshape(outputs, :, 1), targets);
printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = printConfusionMatrix(outputs.>=threshold,    targets);

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs));
    printConfusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;    



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indices = Array{Int64,1}(undef, length(targets));
    indices[  targets] = crossvalidation(sum(  targets), k);
    indices[.!targets] = crossvalidation(sum(.!targets), k);
    return indices;
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    numClasses = size(targets,2);
    @assert(numClasses!=2);
    if numClasses==1
        return crossvalidation(vec(targets), k);
    end;
    indices = Array{Int64,1}(undef, size(targets,1));
    for numClass in 1:numClasses
        indices[targets[:,numClass]] = crossvalidation(sum(targets[:,numClass]), k);
    end;
    return indices;
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    classes = unique(targets);
    numClasses = length(classes);
    indices = Array{Int64,1}(undef, length(targets));
    for class in classes
        indicesThisClass = (targets .== class);
        indices[indicesThisClass] = crossvalidation(sum(indicesThisClass), k);
    end;
    return indices;
end;

# Mejor implementacion:
# crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64) = crossvalidation(oneHotEncoding(targets), k)






function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));

    # Creamos los vectores para las metricas que se vayan a usar
    numFolds = maximum(crossValidationIndices);
    testAccuracy    = Array{Float64,1}(undef, numFolds);
    testErrorRate   = Array{Float64,1}(undef, numFolds);
    testRecall      = Array{Float64,1}(undef, numFolds);
    testSpecificity = Array{Float64,1}(undef, numFolds);
    testPrecision   = Array{Float64,1}(undef, numFolds);
    testNPV         = Array{Float64,1}(undef, numFolds);
    testF1          = Array{Float64,1}(undef, numFolds);

    # Que clases de salida tenemos
    classes = unique(targets);

    # Primero codificamos las salidas deseadas en caso de entrenar RR.NN.AA.
    targets = oneHotEncoding(targets, classes);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Dividimos los datos en entrenamiento y test
        trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
        testInputs        = inputs[crossValidationIndices.==numFold,:];
        trainingTargets   = targets[crossValidationIndices.!=numFold,:];
        testTargets       = targets[crossValidationIndices.==numFold,:];

        # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
        #  se crean vectores adicionales para almacenar las metricas para cada entrenamiento
        testAccuracyEachRepetition    = Array{Float64,1}(undef, numExecutions);
        testErrorRateEachRepetition   = Array{Float64,1}(undef, numExecutions);
        testRecallEachRepetition      = Array{Float64,1}(undef, numExecutions);
        testSpecificityEachRepetition = Array{Float64,1}(undef, numExecutions);
        testPrecisionEachRepetition   = Array{Float64,1}(undef, numExecutions);
        testNPVEachRepetition         = Array{Float64,1}(undef, numExecutions);
        testF1EachRepetition          = Array{Float64,1}(undef, numExecutions);

        # Se entrena las veces que se haya indicado
        for numTraining in 1:numExecutions

            if validationRatio>0

                # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                #  Para ello, hacemos un hold out
                (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*size(inputs,1)/size(trainingInputs,1));
                # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

                # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                ann, = trainClassANN(topology, (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                    validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                    testDataset =       (testInputs,                          testTargets);
                    transferFunctions = transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal);
                    
            else

                # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                #  teniendo cuidado de codificar las salidas deseadas correctamente
                ann, = trainClassANN(topology, (trainingInputs, trainingTargets),
                    testDataset = (testInputs,     testTargets);
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate);

            end;

            # Calculamos las metricas correspondientes con la funcion desarrollada en el ejercicio anterior
            (testAccuracyEachRepetition[numTraining], testErrorRateEachRepetition[numTraining], testRecallEachRepetition[numTraining], testSpecificityEachRepetition[numTraining], testPrecisionEachRepetition[numTraining], testNPVEachRepetition[numTraining], testF1EachRepetition[numTraining], _) =
                confusionMatrix(collect(ann(testInputs')'), testTargets);

        end;

        # Almacenamos las metricas como una media de las obtenidas en los entrenamientos de este fold
        testAccuracy[numFold]    = mean(testAccuracyEachRepetition);
        testErrorRate[numFold]   = mean(testErrorRateEachRepetition);
        testRecall[numFold]      = mean(testRecallEachRepetition);
        testSpecificity[numFold] = mean(testSpecificityEachRepetition);
        testPrecision[numFold]   = mean(testPrecisionEachRepetition);
        testNPV[numFold]         = mean(testNPVEachRepetition);
        testF1[numFold]          = mean(testF1EachRepetition);

    end; # for numFold in 1:numFolds

    return (mean(testAccuracy), std(testAccuracy)), (mean(testErrorRate), std(testErrorRate)), (mean(testRecall), std(testRecall)), (mean(testSpecificity), std(testSpecificity)), (mean(testPrecision), std(testPrecision)), (mean(testNPV), std(testNPV)), (mean(testF1), std(testF1));

end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using ScikitLearn: @sk_import, fit!, predict

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier



function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));

    # Vamos a usar RR.NN.AA.
    if (modelType==:ANN)

        return ANNCrossValidation(modelHyperparameters["topology"],
            inputs, targets, crossValidationIndices;
            numExecutions     = haskey(modelHyperparameters, "numExecutions")     ? modelHyperparameters["numExecutions"]     : 50,
            transferFunctions = haskey(modelHyperparameters, "transferFunctions") ? modelHyperparameters["transferFunctions"] : fill(σ, length(modelHyperparameters["topology"])),
            maxEpochs         = haskey(modelHyperparameters, "maxEpochs")         ? modelHyperparameters["maxEpochs"]         : 1000,
            minLoss           = haskey(modelHyperparameters, "minLoss")           ? modelHyperparameters["minLoss"]           : 0.0,
            learningRate      = haskey(modelHyperparameters, "learningRate")      ? modelHyperparameters["learningRate"]      : 0.01,
            validationRatio   = haskey(modelHyperparameters, "validationRatio")   ? modelHyperparameters["validationRatio"]   : 0,
            maxEpochsVal      = haskey(modelHyperparameters, "maxEpochsVal")      ? modelHyperparameters["maxEpochsVal"]      : 20);

	# Otra posibilidad similar:
#        return ANNCrossValidation(modelHyperparameters["topology"],
#            inputs, targets, crossValidationIndices;
#            numExecutions     = get(modelHyperparameters, "numExecutions",     50,
#            transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(modelHyperparameters["topology"])),
#            maxEpochs         = get(modelHyperparameters, "maxEpochs",         1000,
#            minLoss           = get(modelHyperparameters, "minLoss",           0.0,
#            learningRate      = get(modelHyperparameters, "learningRate",      0.01,
#            validationRatio   = get(modelHyperparameters, "validationRatio",   0,
#            maxEpochsVal      = get(modelHyperparameters, "maxEpochsVal",      20);

    end;

    # No estamos trabajando con redes de neuronas

    # Creamos los vectores para las metricas que se vayan a usar
    numFolds = maximum(crossValidationIndices);
    testAccuracy    = Array{Float64,1}(undef, numFolds);
    testErrorRate   = Array{Float64,1}(undef, numFolds);
    testRecall      = Array{Float64,1}(undef, numFolds);
    testSpecificity = Array{Float64,1}(undef, numFolds);
    testPrecision   = Array{Float64,1}(undef, numFolds);
    testNPV         = Array{Float64,1}(undef, numFolds);
    testF1          = Array{Float64,1}(undef, numFolds);

    targets = string.(targets);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Dividimos los datos en entrenamiento y test
        trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
        testInputs        = inputs[crossValidationIndices.==numFold,:];
        trainingTargets   = targets[crossValidationIndices.!=numFold];
        testTargets       = targets[crossValidationIndices.==numFold];

        # Creamos el modelo según lo que nos hayan pasado como parámetro
        if modelType==:SVC
            @assert((modelHyperparameters["kernel"] == "linear") || (modelHyperparameters["kernel"] == "poly") || (modelHyperparameters["kernel"] == "rbf") || (modelHyperparameters["kernel"] == "sigmoid"));
            # model = SVC(C=modelHyperparameters["C"], kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["degree"], gamma=modelHyperparameters["gamma"], coef0=modelHyperparameters["coef0"]);
            if(modelHyperparameters["kernel"] == "linear")
                model = SVC(kernel=modelHyperparameters["kernel"], C=modelHyperparameters["C"]);
            elseif(modelHyperparameters["kernel"] == "poly")
                model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["degree"], gamma=modelHyperparameters["gamma"], C=modelHyperparameters["C"], coef0=modelHyperparameters["coef0"]);
            elseif(modelHyperparameters["kernel"] == "rbf")
                model = SVC(kernel=modelHyperparameters["kernel"], gamma=modelHyperparameters["gamma"], C=modelHyperparameters["C"]);
            elseif(modelHyperparameters["kernel"] == "sigmoid")
                model = SVC(kernel=modelHyperparameters["kernel"], gamma=modelHyperparameters["gamma"], C=modelHyperparameters["C"], coef0=modelHyperparameters["coef0"]);
            else
                error(string("Unknown kernel; ", kernel));
            end;
        elseif modelType==:DecisionTreeClassifier
            model = DecisionTreeClassifier(max_depth=modelHyperparameters["max_depth"], random_state=1);
        elseif modelType==:KNeighborsClassifier
            model = KNeighborsClassifier(modelHyperparameters["n_neighbors"]);
        else
            error(string("Unknown model ", modelType));
        end;

        # Entrenamos el modelo con el conjunto de entrenamiento
        model = fit!(model, trainingInputs, trainingTargets);

        # Pasamos el conjunto de test
        testOutputs = predict(model, testInputs);

        # Calculamos las metricas y las almacenamos en las posiciones de este fold de cada vector

        (testAccuracy[numFold], testErrorRate[numFold], testRecall[numFold], testSpecificity[numFold], testPrecision[numFold], testNPV[numFold], testF1[numFold], _) =
        confusionMatrix(testOutputs, testTargets);

    end; # for numFold in 1:numFolds

    return (mean(testAccuracy), std(testAccuracy)), (mean(testErrorRate), std(testErrorRate)), (mean(testRecall), std(testRecall)), (mean(testSpecificity), std(testSpecificity)), (mean(testPrecision), std(testPrecision)), (mean(testNPV), std(testNPV)), (mean(testF1), std(testF1));

end;


function manual_undersample(features, targets)
    counts = countmap(targets)
    min_count = minimum(values(counts))
    
    indices = Int[]
    for class in keys(counts)
        class_indices = findall(t -> t == class, targets)
        sampled_indices = sample(class_indices, min_count, replace=false)
        append!(indices, sampled_indices)
    end
    
    return features[indices, :], targets[indices]
end

function get_column_data(results, index)
    if index == 1
        return [result[1] for result in results]
    elseif index == 2
        return [result[2] for result in results]
    elseif index == 3
        return [result[3][1] for result in results]
    elseif index == 4
        return [result[3][2] for result in results]
    end
end