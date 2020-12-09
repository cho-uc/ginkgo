/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/factorization/par_ilu_kernels.hpp"


#include <ginkgo/core/matrix/coo.hpp>


#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/merging.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The parallel ilu factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilu_factorization {


constexpr int default_block_size{512};


// subwarp sizes for all warp-parallel kernels (sweep)
using compiled_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/factorization/par_ilu_kernels.hpp.inc"


namespace {


template <int subwarp_size, typename ValueType, typename IndexType>
void compute_l_u_factors(syn::value_list<int, subwarp_size>,
                         std::shared_ptr<const CudaExecutor> exec,
                         size_type iterations,
                         const matrix::Coo<ValueType, IndexType> *system_matrix,
                         matrix::Csr<ValueType, IndexType> *l_factor,
                         matrix::Csr<ValueType, IndexType> *u_factor)
{
    auto total_nnz =
        static_cast<IndexType>(system_matrix->get_num_stored_elements());
    auto block_size = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(total_nnz, block_size);
    for (size_type i = 0; i < iterations; ++i) {
        kernel::compute_l_u_factors_subwarp<subwarp_size>
            <<<num_blocks, default_block_size, 0, 0>>>(
                system_matrix->get_num_stored_elements(),
                system_matrix->get_const_row_idxs(),
                system_matrix->get_const_col_idxs(),
                as_cuda_type(system_matrix->get_const_values()),
                l_factor->get_const_row_ptrs(), l_factor->get_const_col_idxs(),
                as_cuda_type(l_factor->get_values()),
                u_factor->get_const_row_ptrs(), u_factor->get_const_col_idxs(),
                as_cuda_type(u_factor->get_values()));
    }
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_l_u_factors,
                                    compute_l_u_factors);


}  // namespace


template <typename ValueType, typename IndexType>
void compute_l_u_factors(std::shared_ptr<const CudaExecutor> exec,
                         size_type iterations,
                         const matrix::Coo<ValueType, IndexType> *system_matrix,
                         matrix::Csr<ValueType, IndexType> *l_factor,
                         matrix::Csr<ValueType, IndexType> *u_factor)
{
    iterations = (iterations == 0) ? 10 : iterations;
    if (l_factor->get_strategy()->get_name() == "classical") {
        const auto num_elements = system_matrix->get_num_stored_elements();
        const dim3 block_size{default_block_size, 1, 1};
        const dim3 grid_dim{
            static_cast<uint32>(
                ceildiv(num_elements, static_cast<size_type>(block_size.x))),
            1, 1};
        for (size_type i = 0; i < iterations; ++i) {
            kernel::compute_l_u_factors<<<grid_dim, block_size, 0, 0>>>(
                num_elements, system_matrix->get_const_row_idxs(),
                system_matrix->get_const_col_idxs(),
                as_cuda_type(system_matrix->get_const_values()),
                l_factor->get_const_row_ptrs(), l_factor->get_const_col_idxs(),
                as_cuda_type(l_factor->get_values()),
                u_factor->get_const_row_ptrs(), u_factor->get_const_col_idxs(),
                as_cuda_type(u_factor->get_values()));
        }
    } else {
        auto work = l_factor->get_num_stored_elements() +
                    u_factor->get_num_stored_elements();
        auto work_per_row = work / system_matrix->get_size()[0];
        select_compute_l_u_factors(
            compiled_kernels(),
            [&](int compiled_subwarp_size) {
                return work_per_row <= compiled_subwarp_size ||
                       compiled_subwarp_size == config::warp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, iterations,
            system_matrix, l_factor, u_factor);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_COMPUTE_L_U_FACTORS_KERNEL);


}  // namespace par_ilu_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
