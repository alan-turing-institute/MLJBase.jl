abstract type MLJTask <: MLJType end # `Task` already has meaning in Base

mutable struct UnsupervisedTask <: MLJTask
    X
    input_scitypes::DataType
    input_is_multivariate::Bool
end

"""
    task = UnsupervisedTask(data=nothing, ignore=Symbol[], input_is_multivariate=true)

Construct an unsupervised learning task with given input `data`, which
should be a table or, in the case of univariate inputs, a single
vector. 

Rows of `data` must correspond to patterns and columns to
features. Columns in `data` whose names appear in `ignore` will be
ignored by models trained on the task.

    X = task()

Return the input data (with ignored features removed). 

"""
function UnsupervisedTask(; data=nothing, ignore=Symbol[], input_is_multivariate=true)

    data != nothing || error("You must specify data=... ")
    
    if ignore isa Symbol
        ignore = [ignore, ]
    end

    features = filter(schema(data).names |> collect) do ftr
        !(ftr in ignore)
    end
    X = selectcols(data, collect(features))
    
    input_scitypes = union_scitypes(X)

    return UnsupervisedTask(X, input_scitypes, input_is_multivariate)
end

# U is true for univariate targest.
# X and y can be different views of a common object.
mutable struct SupervisedTask{U} <: MLJTask 
    X                               
    y
    is_probabilistic
    target_scitype
    input_scitypes
    input_is_multivariate::Bool
end

"""
    task = SupervisedTask(X, y; is_probabilistic=nothing, input_is_multivariate=true, target_is_multivariate=false)

Construct a supervised learning task with input features `X` and
target `y`. Both `X` and `y` must be tables or vectors, according to
whether they are multivariate or univariate. Table rows must
correspond to patterns and columns to features. The boolean keyword
argument `is_probabilistic` must be specified.

    task = SupervisedTask(data=nothing, is_probabilistic=nothing, targets=nothing, ignore=Symbol[], input_is_multivariate=true)

Construct a supervised learning task with input features `X` and
target `y`, where `y` is the column vector from `data` named `target`
(if this is a single symbol) or, a table whose columns are those named
in `target` (if this is vector); `X` consists of all remaining columns
of `data` not named in `ignore`.

    X, y = task()

Returns the input `X` and target `y` of the task, also available as
`task.X` and `task.y`.

"""
function SupervisedTask(X, y; is_probabilistic=nothing, input_is_multivariate=true, target_is_multivariate=false)

    is_probabilistic != nothing ||
        error("You must specify is_probabilistic=true or is_proabilistic=false. ")

    if target_is_multivariate
        target_scitype = Tuple{[union_scitypes(selectcols(y, c)) for c in schema(y).names]...}
    else
        target_scitype = union_scitypes(y)
    end

    input_scitypes = union_scitypes(X)

    return SupervisedTask{!target_is_multivariate}(X, y, is_probabilistic, target_scitype,
                                         input_scitypes, input_is_multivariate)
end

function SupervisedTask(; data=nothing, is_probabilistic=nothing, target=nothing, ignore=Symbol[], input_is_multivariate=true)

    data != nothing ||
        error("You must specify data=... or use SupervisedTask(X, y, ...).")

    is_probabilistic != nothing ||
        error("You must specify is_probabilistic=true or is_proabilistic=false. ")

    if ignore isa Symbol
        ignore = [ignore, ]
    end
    
    target != nothing ||
        error("You must specify target=... (use Symbol or Vector{Symbol}) ")

    if target isa Vector
        target_is_multivariate = true
        issubset(Set(target), Set(schema(data).names)) ||
                error("One or more specified targets missing from supplied data. ")
        y = selectcols(data, target)
    else
        target_is_multivariate = false
        target in schema(data).names ||
            error("Specificed target not found in data. ")
        y = selectcols(data, target)
        target = [target, ]
    end

    features = filter(schema(data).names |> collect) do ftr
        !(ftr in target) && !(ftr in ignore)
    end

    if length(features) == 1
        X = selectcols(data, features[1])
        input_is_multivariate = false
    else
        X = selectcols(data, features)
        input_is_multivariate = true
    end
    
    return SupervisedTask(X, y; is_probabilistic=is_probabilistic,
                          input_is_multivariate=input_is_multivariate,
                          target_is_multivariate=target_is_multivariate)
end



## RUDIMENTARY TASK OPERATIONS

nrows(task::MLJTask) = nrows(task.X)
Base.eachindex(task::MLJTask) = Base.OneTo(nrows(task))
nfeatures(task::MLJTask) = length(schema(task.X).names)

X_(task::MLJTask) = task.X
y_(task::SupervisedTask) = task.y

X_and_y(task::SupervisedTask) = (X_(task), y_(task))

# make tasks callable:
(task::UnsupervisedTask)() = X_(task)
(task::SupervisedTask)() = X_and_y(task)

