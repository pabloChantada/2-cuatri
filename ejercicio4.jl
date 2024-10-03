using Flux;
using Flux.Losses;
using FileIO;
using JLD2;
using Images;
using DelimitedFiles;
using Test;
using Statistics
using LinearAlgebra;


Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}} 

function batchInputs(batch::Batch) 
    batchInputs = batch[1]

    return batchIntputs
end;


function batchTargets(batch::Batch) 
    batchTargets = batch[2]
    return batchTargets
end;


function batchLength(batch::Batch) 
    batchInputs = batchInputs(batch)
    lenght = size(batchInputs, 1)
    return lenght
end;