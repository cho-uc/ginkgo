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
#include <iostream>


template <typename MtxType>
std::shared_ptr<gko::solver::Cg<typename MtxType::value_type>> gen_solver(
    std::shared_ptr<const gko::Executor> exec, std::shared_ptr<MtxType> mtx)
{
    using value_type = typename MtxType::value_type;
    const gko::remove_complex<value_type> reduction_factor{1e-12};
    using precond =
        gko::preconditioner::Ilu<gko::solver::LowerTrs<value_type>,
                                 gko::solver::UpperTrs<value_type>, false>;
    std::shared_ptr<typename precond::Factory> ilu_precond =
        precond::build().on(exec);
    auto solver_gen =
        gko::solver::Cg<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2000u).on(exec),
                gko::stop::ResidualNormReduction<value_type>::build()
                    .with_reduction_factor(reduction_factor)
                    .on(exec))
            .with_preconditioner(ilu_precond)
            .on(exec);
    return solver_gen->generate(mtx);
}

int main(int argc, char *argv[])
{
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;

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

    // IF mtx is on host, it works!
    auto mtx_device = share(clone(A));
    // IF mtx is on device, it fails!
    // auto mtx_device = share(clone(exec, A));
    auto solver = gen_solver(exec, mtx_device);

    solver->apply(lend(b), lend(x));

    // Print the solution to the command line.
    std::cout << "Solution (x):\n";
    // write(std::cout, lend(x));

    auto one = gko::initialize<vec>({1.0}, host);
    auto neg_one = gko::initialize<vec>({-1.0}, host);
    auto res = gko::initialize<real_vec>({0.0}, host);
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(res));

    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, lend(res));
}
