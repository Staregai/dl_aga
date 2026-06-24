using Images
using FileIO
using StatsBase
using Flux
using OneHotArrays
using Random

project_root = dirname(@__DIR__)
data_dir = joinpath(project_root, "data", "raw", "kaggle_room_street_data", "house_data")
@assert isdir(data_dir) "Missing dataset folder: $data_dir"

all_files = filter(f -> endswith(lowercase(f), ".jpg"), readdir(data_dir))
class_from_filename(filename::String) = split(filename, "_")[1]
labels = [class_from_filename(f) for f in all_files]
counts = countmap(labels)

println("Total house_data JPG files: ", length(all_files))
println("Class counts: ", sort(collect(counts)))

valid_indices = findall(x -> x != "bath", labels)
filtered_filepaths = [joinpath(data_dir, all_files[i]) for i in valid_indices]
filtered_labels = labels[valid_indices]
unique_labels = unique(filtered_labels)
label_to_id = Dict(label => i for (i, label) in enumerate(unique_labels))

const IMG_SIZE = (64, 64)

function process_image(path)
    img = load(path)
    img_resized = imresize(img, IMG_SIZE)
    arr = Float32.(channelview(img_resized))

    if ndims(arr) == 2 || size(arr, 1) != 3
        return nothing
    end

    return permutedims(arr, (3, 2, 1))
end

sample_count = min(8, length(filtered_filepaths))
tensors = process_image.(filtered_filepaths[1:sample_count])
@assert all(t -> t !== nothing && size(t) == (64, 64, 3), tensors)

X = Array{Float32, 4}(undef, 64, 64, 3, sample_count)
for i in 1:sample_count
    X[:, :, :, i] .= tensors[i]
end
Y = onehotbatch([label_to_id[l] for l in filtered_labels[1:sample_count]], 1:length(unique_labels))

model = Chain(
    Conv((3, 3), 3 => 4, relu, pad=1),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(32 * 32 * 4 => length(unique_labels)),
)

println("Labels after bath removal: ", unique_labels)
println("X shape: ", size(X))
println("Y shape: ", size(Y))
println("Mini-CNN output shape: ", size(model(X)))
