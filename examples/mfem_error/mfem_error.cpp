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

#include <ginkgo/ginkgo.hpp>


#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>


/*
MFEM build command:
cmake ../                       \
    -DCMAKE_BUILD_TYPE=Release  \
    -DMFEM_USE_CUDA=YES         \
    -DCUDA_ARCH="sm_75"         \
    -DMFEM_USE_HIP=NO           \
    -DMFEM_USE_GINKGO=ON        \
    -DCMAKE_BUILD_TYPE=Release  \
    -G "Ninja"                  \
    -DCMAKE_PREFIX_PATH="/path/to/ginkgo/install_dir"
*/


template <typename Precond, typename MtxType>
std::shared_ptr<gko::solver::Cg<typename MtxType::value_type>> gen_solver(
    std::shared_ptr<const gko::Executor> exec, std::shared_ptr<MtxType> mtx)
{
    using value_type = typename MtxType::value_type;
    using index_type = typename MtxType::index_type;
    const gko::remove_complex<value_type> reduction_factor{1e-12};
    using precond = Precond;
    /*
    std::shared_ptr<gko::LinOpFactory> factorization;
    if (exec == exec->get_master()) {
        factorization =
            share(gko::factorization::ParIlu<value_type, index_type>::build()
                      .with_iterations(100u)
                      .on(exec));
    } else {
        factorization = share(
            gko::factorization::Ilu<value_type, index_type>::build().on(exec));
    }
    std::shared_ptr<typename precond::Factory> ilu_precond =
        precond::build().with_factorization_factory(factorization).on(exec);
    */

    auto bj_precond = share(
        precond::build()
            .with_max_block_size(16u)
            .with_storage_optimization(gko::precision_reduction::autodetect())
            .on(exec));
    auto solver_gen =
        gko::solver::Cg<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2000u).on(exec),
                gko::stop::ResidualNormReduction<value_type>::build()
                    .with_reduction_factor(reduction_factor)
                    .on(exec))
            .with_preconditioner(bj_precond)
            .on(exec);
    return solver_gen->generate(mtx);
}


template <typename MtxType, typename ValueType>
void validate_result(const MtxType *mtx, const gko::matrix::Dense<ValueType> *b,
                     const gko::matrix::Dense<ValueType> *x)
{
    using value_type = ValueType;
    using vec = gko::matrix::Dense<value_type>;
    using real_vec = gko::matrix::Dense<gko::remove_complex<value_type>>;
    const auto exec = b->get_executor();
    auto b_clone = clone(b);
    // Print the solution to the command line.
    std::cout << "Solution (x):\n";
    // write(std::cout, lend(x));

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    mtx->apply(lend(one), x, lend(neg_one), b_clone.get());
    b_clone->compute_norm2(lend(res));

    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, lend(res));
}


template <typename CsrType>
void compare(const CsrType *a, const CsrType *b, double delta = 1e-1)
{
    auto ha = clone(a->get_executor()->get_master(), a);
    auto hb = clone(b->get_executor()->get_master(), b);

    auto a_sz = a->get_size();
    auto b_sz = b->get_size();
    auto a_elms = ha->get_num_stored_elements();
    const auto a_val = ha->get_const_values();
    const auto a_col = ha->get_const_col_idxs();
    const auto a_rp = ha->get_const_row_ptrs();
    auto b_elms = hb->get_num_stored_elements();
    const auto b_val = hb->get_const_values();
    const auto b_col = hb->get_const_col_idxs();
    const auto b_rp = hb->get_const_row_ptrs();

    auto is_diff = [delta](double x, double y) {
        using std::abs;
        using std::max;

        auto mn = max(abs(x), abs(y));
        return mn == 0. ? false : (abs(x - y) / mn > delta);
    };

    if (a_sz != b_sz) {
        std::cerr << "Mismatching size!\n";
        return;
    } else if (a_elms != b_elms) {
        std::cerr << "Mismatching number of elements!\n";
        return;
    }
    std::ios_base::fmtflags flag(std::cout.flags());

    std::cout << std::scientific;
    std::cout << std::setprecision(16);

    using size_type = decltype(a_elms);

    auto print_idx = [&](size_type idx, size_type row) {
        std::cout << "A[" << row << ", " << a_col[idx] << "] = " << a_val[idx]
                  << '\n';
        std::cout << "B[" << row << ", " << b_col[idx] << "] = " << b_val[idx]
                  << '\n';
    };

    for (size_type row = 0; row < a_sz[0]; ++row) {
        if (a_rp[row] != b_rp[row]) {
            std::cerr << "Mismatching row ptrs:\n"
                      << "\tA_rp[" << row << "] = " << a_rp[row] << "\n\tB_rp["
                      << row << "] = " << b_rp[row] << '\n';
        }
        for (size_type idx = a_rp[row]; idx < a_rp[row + 1]; ++idx) {
            if (is_diff(a_val[idx], b_val[idx]) || a_col[idx] != b_col[idx]) {
                print_idx(idx, row);
            }
        }
    }

    std::cout.flags(flag);
}


int main(int argc, char *argv[])
{
    using value_type = double;
    using real_value_type = gko::remove_complex<value_type>;
    using index_type = int;
    using vec = gko::matrix::Dense<value_type>;
    using real_vec = gko::matrix::Dense<real_value_type>;
    using mtx = gko::matrix::Csr<value_type, index_type>;
    using cg = gko::solver::Cg<value_type>;
    using precond = gko::preconditioner::Jacobi<value_type, index_type>;
    /*
    using precond =
        gko::preconditioner::Ilu<gko::solver::LowerTrs<value_type>,
                                 gko::solver::UpperTrs<value_type>, false>;
    */

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    auto ref = gko::ReferenceExecutor::create();
    auto omp = gko::OmpExecutor::create();
    auto host = ref;
    auto cuda = gko::CudaExecutor::create(0, host);

    auto exec = cuda;

    auto A = share(gko::read<mtx>(std::ifstream("data/A_o2_beam.mtx"), host));
    A->sort_by_column_index();
    auto b = gko::read<vec>(std::ifstream("data/b_o2_beam.mtx"), host);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), host);
    auto dx = vec::create(host);
    dx->copy_from(x.get());

    // IF mtx is on host, it works!
    // auto mtx_device = share(clone(A));
    // IF mtx is on device, it fails!
    auto hsolver = gen_solver<precond>(host, A);
    hsolver->apply(lend(b), lend(x));
    validate_result(A.get(), b.get(), x.get());

    auto mtx_device = share(clone(cuda, A));
    auto dsolver = gen_solver<precond>(cuda, mtx_device);
    dsolver->apply(lend(b), lend(dx));
    validate_result(A.get(), b.get(), dx.get());

    /*
    auto extract_l_mtx = [](const cg *solver) {
        return static_cast<const precond *>(solver->get_preconditioner().get())
            ->get_l_solver()
            ->get_system_matrix();
    };
    auto extract_u_mtx = [](const cg *solver) {
        return static_cast<const precond *>(solver->get_preconditioner().get())
            ->get_u_solver()
            ->get_system_matrix();
    };
    auto hL = extract_l_mtx(hsolver.get());
    auto hU = extract_u_mtx(hsolver.get());

    auto dL = extract_l_mtx(dsolver.get());
    auto dU = extract_u_mtx(dsolver.get());

    // compare(hL.get(), dL.get());
    */
}
