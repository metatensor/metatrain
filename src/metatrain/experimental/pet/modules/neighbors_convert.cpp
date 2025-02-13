// #include <torch/extension.h>
#include <vector>
#include <algorithm>  // For std::fill
#include <c10/util/Optional.h>  // For c10::optional
#include <torch/script.h>
#include <optional>

// Template function to process the neighbors
template <typename int_t, typename float_t>
std::vector<at::Tensor> process_neighbors_cpu(at::Tensor i_list, at::Tensor j_list, at::Tensor S_list, at::Tensor D_list, 
                                                        int64_t max_size, int64_t n_atoms, at::Tensor species,
                                                        at::Tensor all_species) {
    // Ensure the tensors are on the CPU and are contiguous
    TORCH_CHECK(i_list.device().is_cpu(), "i_list must be on CPU");
    TORCH_CHECK(j_list.device().is_cpu(), "j_list must be on CPU");
    TORCH_CHECK(S_list.device().is_cpu(), "S_list must be on CPU");
    TORCH_CHECK(D_list.device().is_cpu(), "D_list must be on CPU");
    TORCH_CHECK(species.device().is_cpu(), "species must be on CPU");
    TORCH_CHECK(all_species.device().is_cpu(), "all_species must be on CPU");

    TORCH_CHECK(i_list.is_contiguous(), "i_list must be contiguous");
    TORCH_CHECK(j_list.is_contiguous(), "j_list must be contiguous");
    TORCH_CHECK(S_list.is_contiguous(), "S_list must be contiguous");
    TORCH_CHECK(D_list.is_contiguous(), "D_list must be contiguous");
    TORCH_CHECK(species.is_contiguous(), "species must be contiguous");
    TORCH_CHECK(all_species.is_contiguous(), "all_species must be contiguous");

    // Ensure the sizes match
    TORCH_CHECK(i_list.sizes() == j_list.sizes(), "i_list and j_list must have the same size");
    TORCH_CHECK(i_list.size(0) == S_list.size(0) && S_list.size(1) == 3, "S_list must have the shape [N, 3]");
    TORCH_CHECK(i_list.size(0) == D_list.size(0) && D_list.sizes() == S_list.sizes(), "D_list must have the same shape as S_list");

    // Initialize tensors with zeros
    auto options_int = torch::TensorOptions().dtype(i_list.dtype()).device(torch::kCPU);
    auto options_float = torch::TensorOptions().dtype(D_list.dtype()).device(torch::kCPU);
    auto options_bool = torch::TensorOptions().dtype(at::kBool).device(torch::kCPU);

    at::Tensor neighbors_index = torch::zeros({n_atoms, max_size}, options_int);
    at::Tensor neighbors_shift = torch::zeros({n_atoms, max_size, 3}, options_int);
    at::Tensor relative_positions = torch::zeros({n_atoms, max_size, 3}, options_float);
    at::Tensor nums = torch::zeros({n_atoms}, options_int);  // Tensor to store the count of elements
    at::Tensor mask = torch::ones({n_atoms, max_size}, options_bool);  // Tensor to store the mask
    at::Tensor neighbor_species = all_species.size(0) * torch::ones({n_atoms, max_size}, options_int);

    int64_t scalar_attr_dim = 0;

    // Temporary array to track the current population index
    int_t* current_index = new int_t[n_atoms];
    std::fill(current_index, current_index + n_atoms, 0);  // Fill the array with zeros

    // Get raw data pointers
    int_t* i_list_ptr = i_list.data_ptr<int_t>();
    int_t* j_list_ptr = j_list.data_ptr<int_t>();
    int_t* S_list_ptr = S_list.data_ptr<int_t>();
    float_t* D_list_ptr = D_list.data_ptr<float_t>();
    int_t* species_ptr = species.data_ptr<int_t>();
    int_t* all_species_ptr = all_species.data_ptr<int_t>();

    int_t* neighbors_index_ptr = neighbors_index.data_ptr<int_t>();
    int_t* neighbors_shift_ptr = neighbors_shift.data_ptr<int_t>();
    float_t* relative_positions_ptr = relative_positions.data_ptr<float_t>();
    int_t* nums_ptr = nums.data_ptr<int_t>();
    bool* mask_ptr = mask.data_ptr<bool>();
    int_t* neighbor_species_ptr = neighbor_species.data_ptr<int_t>();
    
    int64_t all_species_size = all_species.size(0);
    
    int_t all_species_maximum = -1;
    for (int64_t k = 0; k < all_species_size; ++k) {
        if (all_species_ptr[k] > all_species_maximum) {
            all_species_maximum = all_species_ptr[k];
        }
    }
    
    int_t* mapping = new int_t[all_species_maximum + 1];
    for (int64_t k = 0; k < all_species_size; ++k) {
        mapping[all_species_ptr[k]] = k;
    }
    
    
   
    // Populate the neighbors_index, neighbors_shift, relative_positions, neighbor_species, and neighbor_scalar_attributes tensors
    
    int64_t shift_i;
    int_t i, j, idx;
    for (int64_t k = 0; k < i_list.size(0); ++k) {
        i = i_list_ptr[k];
        j = j_list_ptr[k];
        idx = current_index[i];
        
        shift_i = i * max_size;
        if (idx < max_size) {
            neighbors_index_ptr[shift_i + idx] = j;
            neighbor_species_ptr[shift_i + idx] = mapping[species_ptr[j]];
            /*for (int64_t q = 0; q < all_species_size; ++q) {
                if (all_species_ptr[q] == species_ptr[j]) {
                    neighbor_species_ptr[i * max_size + idx] = q;
                    break;
                }
            }*/
            
            // Unroll the loop for better computational efficiency
            neighbors_shift_ptr[(shift_i + idx) * 3 + 0] = S_list_ptr[k * 3 + 0];
            neighbors_shift_ptr[(shift_i + idx) * 3 + 1] = S_list_ptr[k * 3 + 1];
            neighbors_shift_ptr[(shift_i + idx) * 3 + 2] = S_list_ptr[k * 3 + 2];

            relative_positions_ptr[(shift_i + idx) * 3 + 0] = D_list_ptr[k * 3 + 0];
            relative_positions_ptr[(shift_i + idx) * 3 + 1] = D_list_ptr[k * 3 + 1];
            relative_positions_ptr[(shift_i + idx) * 3 + 2] = D_list_ptr[k * 3 + 2];

            mask_ptr[shift_i + idx] = false;

            current_index[i]++;
        }
    }

    // Copy current_index to nums
    for (int64_t i = 0; i < n_atoms; ++i) {
        nums_ptr[i] = current_index[i];
    }
    
    at::Tensor neighbors_pos = torch::zeros({n_atoms, max_size}, options_int);
    int_t* neighbors_pos_ptr = neighbors_pos.data_ptr<int_t>();

    // Temporary array to track the current population index
    int_t* current_index_two = new int_t[n_atoms];
    std::fill(current_index_two, current_index_two + n_atoms, 0);  // Fill the array with zeros
    
    int64_t shift_j;
    for (int64_t k = 0; k < i_list.size(0); ++k) {
        i = i_list_ptr[k];
        j = j_list_ptr[k];
        shift_j = j * max_size;
        for (int64_t q = 0; q < current_index[j]; ++q) {
            if (neighbors_index_ptr[shift_j + q] == i && neighbors_shift_ptr[(shift_j + q) * 3 + 0] == -S_list_ptr[k * 3 + 0] && neighbors_shift_ptr[(shift_j + q) * 3 + 1] == -S_list_ptr[k * 3 + 1] && neighbors_shift_ptr[(shift_j + q) * 3 + 2] == -S_list_ptr[k * 3 + 2]) {
                neighbors_pos_ptr[i * max_size + current_index_two[i]] = q;
                current_index_two[i]++;
                break;
            }
        }
    }

    // Clean up temporary memory
    delete[] current_index;
    delete[] current_index_two;
    
    at::Tensor species_mapped = torch::zeros({n_atoms}, options_int);
    int_t* species_mapped_ptr = species_mapped.data_ptr<int_t>();
    for (int64_t k = 0; k < n_atoms; ++k) {
        species_mapped_ptr[k] = mapping[species_ptr[k]];
    }
    
    /*for (int64_t k = 0; k < n_atoms; ++k) {
         for (int64_t q = 0; q < all_species_size; ++q) {
            if (all_species_ptr[q] == species_ptr[k]) {
                species_mapped_ptr[k] = q;
                break;
            }
        }
    }*/
    
     delete[] mapping;
    
     return {neighbors_index, relative_positions, nums, mask, neighbor_species, neighbors_pos, species_mapped};
}

// Template function for backward pass
template <typename int_t, typename float_t>
at::Tensor process_neighbors_cpu_backward(at::Tensor grad_output, at::Tensor i_list, int64_t max_size, int64_t n_atoms) {
    // Ensure the tensors are on the CPU and are contiguous
    TORCH_CHECK(grad_output.device().is_cpu(), "grad_output must be on CPU");
    TORCH_CHECK(i_list.device().is_cpu(), "i_list must be on CPU");

    grad_output = grad_output.contiguous();
    i_list = i_list.contiguous();
    
    // TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    // TORCH_CHECK(i_list.is_contiguous(), "i_list must be contiguous");

    // Initialize gradient tensor for D_list with zeros
    auto options_float = torch::TensorOptions().dtype(grad_output.dtype()).device(torch::kCPU);
    at::Tensor grad_D_list = torch::zeros({i_list.size(0), 3}, options_float);

    int_t* current_index = new int_t[n_atoms];
    std::fill(current_index, current_index + n_atoms, 0);  // Fill the array with zeros

    float_t* grad_D_list_ptr = grad_D_list.data_ptr<float_t>();
    float_t* grad_output_ptr = grad_output.data_ptr<float_t>();
    int_t* i_list_ptr = i_list.data_ptr<int_t>();
    int_t i, idx;

    for (int64_t k = 0; k < i_list.size(0); ++k) {
        i = i_list_ptr[k];
        idx = current_index[i];
        grad_D_list_ptr[k * 3 + 0] = grad_output_ptr[(i * max_size + idx) * 3 + 0];
        grad_D_list_ptr[k * 3 + 1] = grad_output_ptr[(i * max_size + idx) * 3 + 1];
        grad_D_list_ptr[k * 3 + 2] = grad_output_ptr[(i * max_size + idx) * 3 + 2];
        current_index[i]++;
    }

    delete[] current_index;
    return grad_D_list;
}

template <typename int_t, typename float_t>
at::Tensor process_neighbors_backward(at::Tensor grad_output, at::Tensor i_list, int64_t max_size, int64_t n_atoms) {
    // Ensure all tensors are on the same device
    auto device = grad_output.device();
    TORCH_CHECK(i_list.device() == device, "i_list must be on the same device as grad_output");

    // Move all tensors to CPU
    auto grad_output_cpu = grad_output.cpu();
    auto i_list_cpu = i_list.cpu();

    // Invoke the CPU version of the function
    auto grad_D_list_cpu = process_neighbors_cpu_backward<int_t, float_t>(grad_output_cpu, i_list_cpu, max_size, n_atoms);

    // Move the gradient tensor back to the initial device
    return grad_D_list_cpu.to(device);
}

// Dispatch function based on tensor types for backward
at::Tensor process_dispatch_backward(at::Tensor grad_output, at::Tensor i_list, int64_t max_size, int64_t n_atoms) {
    if (i_list.scalar_type() == at::ScalarType::Int && grad_output.scalar_type() == at::ScalarType::Float) {
        return process_neighbors_backward<int32_t, float>(grad_output, i_list, max_size, n_atoms);
    } else if (i_list.scalar_type() == at::ScalarType::Int && grad_output.scalar_type() == at::ScalarType::Double) {
        return process_neighbors_backward<int32_t, double>(grad_output, i_list, max_size, n_atoms);
    } else if (i_list.scalar_type() == at::ScalarType::Long && grad_output.scalar_type() == at::ScalarType::Float) {
        return process_neighbors_backward<int64_t, float>(grad_output, i_list, max_size, n_atoms);
    } else if (i_list.scalar_type() == at::ScalarType::Long && grad_output.scalar_type() == at::ScalarType::Double) {
        return process_neighbors_backward<int64_t, double>(grad_output, i_list, max_size, n_atoms);
    } else {
        throw std::runtime_error("Unsupported tensor types");
    }
}

template <typename int_t, typename float_t>
std::vector<at::Tensor> process_neighbors(at::Tensor i_list, at::Tensor j_list, at::Tensor S_list, at::Tensor D_list, 
                                                        int64_t max_size, int64_t n_atoms, at::Tensor species,
                                                        at::Tensor all_species) {
    // Ensure all tensors are on the same device
    auto device = i_list.device();
    TORCH_CHECK(j_list.device() == device, "j_list must be on the same device as i_list");
    TORCH_CHECK(S_list.device() == device, "S_list must be on the same device as i_list");
    TORCH_CHECK(D_list.device() == device, "D_list must be on the same device as i_list");
    TORCH_CHECK(species.device() == device, "species must be on the same device as i_list");
    TORCH_CHECK(all_species.device() == device, "all_species must be on the same device as i_list");
    
    // Move all tensors to CPU
    auto i_list_cpu = i_list.cpu();
    auto j_list_cpu = j_list.cpu();
    auto S_list_cpu = S_list.cpu();
    auto D_list_cpu = D_list.cpu();
    auto species_cpu = species.cpu();
    auto all_species_cpu = all_species.cpu();

    // Invoke the CPU version of the function
    auto result = process_neighbors_cpu<int_t, float_t>(i_list_cpu, j_list_cpu, S_list_cpu, D_list_cpu, max_size, n_atoms, species_cpu, all_species_cpu);

    // Move the output tensors back to the initial device
    for (auto& tensor_opt : result) {
        tensor_opt = tensor_opt.to(device);
    }

    return result;
}

// Dispatch function based on tensor types
std::vector<at::Tensor> process_dispatch(at::Tensor i_list, at::Tensor j_list, at::Tensor S_list, at::Tensor D_list, 
                                                        int64_t max_size, int64_t n_atoms, at::Tensor species,
                                                        at::Tensor all_species) {
    if (i_list.scalar_type() == at::ScalarType::Int && j_list.scalar_type() == at::ScalarType::Int &&
        S_list.scalar_type() == at::ScalarType::Int && D_list.scalar_type() == at::ScalarType::Float) {
        return process_neighbors<int32_t, float>(i_list, j_list, S_list, D_list, max_size, n_atoms, species, all_species);
    } else if (i_list.scalar_type() == at::ScalarType::Int && j_list.scalar_type() == at::ScalarType::Int &&
               S_list.scalar_type() == at::ScalarType::Int && D_list.scalar_type() == at::ScalarType::Double) {
        return process_neighbors<int32_t, double>(i_list, j_list, S_list, D_list, max_size, n_atoms, species, all_species);
    } else if (i_list.scalar_type() == at::ScalarType::Long && j_list.scalar_type() == at::ScalarType::Long &&
               S_list.scalar_type() == at::ScalarType::Long && D_list.scalar_type() == at::ScalarType::Float) {
        return process_neighbors<int64_t, float>(i_list, j_list, S_list, D_list, max_size, n_atoms, species, all_species);
    } else if (i_list.scalar_type() == at::ScalarType::Long && j_list.scalar_type() == at::ScalarType::Long &&
               S_list.scalar_type() == at::ScalarType::Long && D_list.scalar_type() == at::ScalarType::Double) {
        return process_neighbors<int64_t, double>(i_list, j_list, S_list, D_list, max_size, n_atoms, species, all_species);
    } else {
        throw std::runtime_error("Unsupported tensor types");
    }
}

class ProcessNeighborsFunction : public torch::autograd::Function<ProcessNeighborsFunction> {
public:
    static std::vector<at::Tensor> forward(torch::autograd::AutogradContext *ctx, at::Tensor i_list, at::Tensor j_list, 
                                           at::Tensor S_list, at::Tensor D_list, int64_t max_size, int64_t n_atoms, 
                                           at::Tensor species, at::Tensor all_species) {
        auto outputs = process_dispatch(i_list, j_list, S_list, D_list, max_size, n_atoms, species, all_species);
        ctx->save_for_backward({i_list});
        ctx->saved_data["max_size"] = max_size;
        ctx->saved_data["n_atoms"] = n_atoms;
        return outputs;
    }

    static std::vector<at::Tensor> backward(torch::autograd::AutogradContext *ctx, std::vector<at::Tensor> grad_outputs) {
        auto i_list = ctx->get_saved_variables()[0];
        auto max_size = ctx->saved_data["max_size"].toInt();
        auto n_atoms = ctx->saved_data["n_atoms"].toInt();

        auto grad_relative_positions = grad_outputs[1];  // Assuming this is the gradient w.r.t relative_positions tensor
        auto grad_D_list = process_dispatch_backward(grad_relative_positions, i_list, max_size, n_atoms);

        return {at::Tensor(), at::Tensor(), at::Tensor(), grad_D_list, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

// Wrapper function to call apply
std::vector<at::Tensor> process_neighbors_apply(at::Tensor i_list, at::Tensor j_list, at::Tensor S_list, at::Tensor D_list, 
                                          int64_t max_size, int64_t n_atoms, at::Tensor species, at::Tensor all_species) {
    return ProcessNeighborsFunction::apply(i_list, j_list, S_list, D_list, max_size, n_atoms, species, all_species);
}

/*TORCH_LIBRARY(neighbors_convert, m) {
    m.def(
        "convert_neighbors(Tensor i_list, Tensor j_list, Tensor S_list, Tensor D_list, int max_size, int n_atoms, Tensor species, Tensor all_species) -> Tensor[]",
         &process_neighbors_apply
    );
}*/

TORCH_LIBRARY(neighbors_convert, m) {
    m.def(
        "process",
         &process_neighbors_apply
    );
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("process_neighbors(Tensor i_list, Tensor j_list, Tensor S_list, Tensor D_list, int max_size, int n_atoms, Tensor species, Tensor all_species) -> Tensor[]", &process_neighbors_apply, "Process neighbors and return tensors, including count tensor, mask, and neighbor_species");
// }
    
/*static auto registry = torch::RegisterOperators()
    .op("neighbors_convert::process(Tensor i_list, Tensor j_list, Tensor S_list, Tensor D_list, int max_size, int n_atoms, Tensor species, Tensor all_species) -> Tensor[]", &process_neighbors_apply);*/

/*PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("process_neighbors", &process_neighbors_apply, "Process neighbors and return tensors, including count tensor, mask, and neighbor_species");*/
    
/*static auto registry = torch::RegisterOperators()
    .op("neighbors_convert::process(Tensor i_list, Tensor j_list, Tensor S_list, Tensor D_list, int max_size, int n_atoms, Tensor species, Tensor all_species) -> Tensor[]", &process_neighbors_apply);

*/