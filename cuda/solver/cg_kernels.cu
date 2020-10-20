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

#include "core/solver/cg_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The CG solver namespace.
 *
 * @ingroup cg
 */
namespace cg {


namespace syn {
template <typename ValueType>
struct scalar_t {
    matrix::Dense<ValueType> *data;
};

template <typename ValueType>
struct vector_t {
    matrix::Dense<ValueType> *data;
};

template <typename ValueType>
struct const_scalar_t {
    const matrix::Dense<ValueType> *data;
};

template <typename ValueType>
struct const_vector_t {
    const matrix::Dense<ValueType> *data;
};

template <typename ValueType>
scalar_t<ValueType> scalar(matrix::Dense<ValueType> *mtx)
{
    return {mtx};
}

template <typename ValueType>
const_scalar_t<ValueType> scalar(const matrix::Dense<ValueType> *mtx)
{
    return {mtx};
}

template <typename ValueType>
vector_t<ValueType> vector(matrix::Dense<ValueType> *mtx)
{
    return {mtx};
}

template <typename ValueType>
const_vector_t<ValueType> vector(const matrix::Dense<ValueType> *mtx)
{
    return {mtx};
}

template <typename ValueType>
struct device_dense {
    cuda_type<ValueType> *__restrict__ data;
    size_type stride;
};

template <typename ValueType>
device_dense<cuda_type<ValueType>> map_to_device(vector_t<ValueType> mtx)
{
    return {as_cuda_type(mtx.data->get_values()), mtx.data->get_stride()};
}

template <typename ValueType>
device_dense<cuda_type<const ValueType>> map_to_device(
    const_vector_t<ValueType> mtx)
{
    return {as_cuda_type(mtx.data->get_const_values()), mtx.data->get_stride()};
}

template <typename ValueType>
struct device_scalar {
    ValueType *__restrict__ data;
};

template <typename ValueType>
device_scalar<cuda_type<ValueType>> map_to_device(scalar_t<ValueType> mtx)
{
    return {as_cuda_type(mtx.data->get_values())};
}

template <typename ValueType>
device_scalar<cuda_type<const ValueType>> map_to_device(
    const_scalar_t<ValueType> mtx)
{
    return {as_cuda_type(mtx.data->get_const_values())};
}

stopping_status *map_to_device(Array<stopping_status> &status)
{
    return status.get_data();
}

template <typename ValueType>
__device__ ValueType &unpack_on_device(size_type row, size_type col,
                                       device_dense<ValueType> mtx)
{
    return mtx.data[row * mtx.stride + col];
}

template <typename ValueType>
__device__ ValueType &unpack_on_device(size_type row, size_type col,
                                       device_scalar<ValueType> mtx)
{
    return mtx.data[col];
}

__device__ stopping_status &unpack_on_device(size_type row, size_type col,
                                             stopping_status *status)
{
    return status[col];
}

template <typename Function, typename... Args>
__global__ void generic_2d_kernel(size_type num_rows, size_type num_cols,
                                  Function func, Args... args)
{
    auto col = threadIdx.x + blockDim.x * blockIdx.x;
    auto row = threadIdx.y + blockDim.y * blockIdx.y;
    if (row < num_rows && col < num_cols) {
        func(unpack_on_device(row, col, args)..., row, col);
    }
}

template <typename T>
struct size_extract_helper {};

template <typename ValueType>
struct size_extract_helper<vector_t<ValueType>> {
    static constexpr bool has_size() { return true; }
    static gko::dim<2> get_size(vector_t<ValueType> v)
    {
        return v.data->get_size();
    }
    static bool is_compatible(gko::dim<2> size, vector_t<ValueType> v)
    {
        return get_size(v) == size;
    }
};

template <typename ValueType>
struct size_extract_helper<const_vector_t<ValueType>> {
    static constexpr bool has_size() { return true; }
    static gko::dim<2> get_size(const_vector_t<ValueType> v)
    {
        return v.data->get_size();
    }
    static bool is_compatible(gko::dim<2> size, const_vector_t<ValueType> v)
    {
        return get_size(v) == size;
    }
};

template <typename ValueType>
struct size_extract_helper<scalar_t<ValueType>> {
    static constexpr bool has_size() { return false; }
    static gko::dim<2> get_size(scalar_t<ValueType> v)
    {
        return v.data->get_size();
    }
    static bool is_compatible(gko::dim<2> size, scalar_t<ValueType> v)
    {
        return get_size(v)[1] == size[1];
    }
};

template <typename ValueType>
struct size_extract_helper<const_scalar_t<ValueType>> {
    static constexpr bool has_size() { return false; }
    static gko::dim<2> get_size(const_scalar_t<ValueType> v)
    {
        return v.data->get_size();
    }
    static bool is_compatible(gko::dim<2> size, const_scalar_t<ValueType> v)
    {
        return get_size(v)[1] == size[1];
    }
};

template <>
struct size_extract_helper<Array<stopping_status>> {
    static constexpr bool has_size() { return false; }
    static gko::dim<2> get_size(Array<stopping_status> &status)
    {
        return {1, status.get_num_elems()};
    }
    static bool is_compatible(gko::dim<2> size, Array<stopping_status> &v)
    {
        return get_size(v)[1] == size[1];
    }
};

bool all() { return true; }

template <typename Arg, typename... Args>
bool all(Arg arg, Args... args)
{
    return arg && all(args...);
}

gko::dim<2> find_first_size() { return gko::dim<2>{}; }

template <typename Arg, typename... Args>
gko::dim<2> find_first_size(Arg arg, Args... args)
{
    if (size_extract_helper<Arg>::has_size()) {
        return size_extract_helper<Arg>::get_size(arg);
    } else {
        return find_first_size(args...);
    }
}

template <typename Function, typename... Args>
void dispatch(Function func, Args... args)
{
    auto size = find_first_size(args...);
    constexpr auto x_blocksize = 32;
    constexpr auto y_blocksize = 32;
    GKO_ASSERT(all(size_extract_helper<Args>::is_compatible(size, args)...));
    auto x_blocks = ceildiv(size[1], x_blocksize);
    auto y_blocks = ceildiv(size[0], y_blocksize);
    auto blocks = dim3(x_blocks, y_blocks);
    auto threads = dim3(x_blocksize, y_blocksize);
    generic_2d_kernel<<<blocks, threads>>>(size[0], size[1], func,
                                           map_to_device(args)...);
}

}  // namespace syn


template <typename ValueType>
void initialize(std::shared_ptr<const CudaExecutor> exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *z, matrix::Dense<ValueType> *p,
                matrix::Dense<ValueType> *q, matrix::Dense<ValueType> *prev_rho,
                matrix::Dense<ValueType> *rho,
                Array<stopping_status> *stop_status)
{
    using syn::scalar;
    using syn::vector;
    syn::dispatch(
        [] __device__(auto &b, auto &r, auto &z, auto &p, auto &q,
                      auto &prev_rho, auto &rho, auto &stop_status,
                      size_type row, size_type col) {
            if (row == 0) {
                rho = zero();
                prev_rho = one();
                stop_status.reset();
            }
            r = b;
            z = p = q = zero();
        },
        vector(b), vector(r), vector(z), vector(p), vector(q), scalar(prev_rho),
        scalar(rho), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const CudaExecutor> exec,
            matrix::Dense<ValueType> *p, const matrix::Dense<ValueType> *z,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho,
            const Array<stopping_status> *stop_status)
{
    using syn::scalar;
    using syn::vector;
    syn::dispatch(
        [] __device__(auto &p, auto &z, auto &rho, auto &prev_rho,
                      auto &stop_status, size_type row, size_type col) {
            if (!stop_status.has_stopped()) {
                auto tmp = prev_rho == zero(prev_rho) ? 0 : rho / prev_rho;
                p = z + tmp * p;
            }
        },
        vector(p), vector(z), scalar(rho), scalar(prev_rho), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const CudaExecutor> exec,
            matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *r,
            const matrix::Dense<ValueType> *p,
            const matrix::Dense<ValueType> *q,
            const matrix::Dense<ValueType> *beta,
            const matrix::Dense<ValueType> *rho,
            const Array<stopping_status> *stop_status)
{
    using syn::scalar;
    using syn::vector;
    syn::dispatch(
        [] __device__(auto &x, auto &r, auto &p, auto &q, auto &beta, auto &rho,
                      auto &stop_status, size_type row, size_type col) {
            if (!stop_status.has_stopped()) {
                auto tmp = beta == zero(beta) ? zero(beta) : rho / beta;
                x += tmp * p;
                r -= tmp * q;
            }
        },
        vector(x), vector(r), vector(p), vector(q), scalar(beta), scalar(rho),
        *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_STEP_2_KERNEL);


}  // namespace cg
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
