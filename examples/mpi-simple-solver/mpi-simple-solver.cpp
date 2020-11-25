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

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the fstream header to read from data from files.
#include <fstream>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <algorithm>
#include <string>


int main(int argc, char *argv[])
{
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    // using Mtx = gko::matrix::Dense<ValueType>;
    using Solver = gko::solver::Cg<ValueType>;
    // The gko::matrix::Csr class is used here, but any other matrix class such
    // as gko::matrix::Coo, gko::matrix::Hybrid, gko::matrix::Ell or
    // gko::matrix::Sellp could also be used.
    using Mtx = gko::matrix::Csr<ValueType, IndexType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    // The gko::solver::Cg is used here, but any other solver class can also be
    // used.
    using cg = gko::solver::Cg<ValueType>;

    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }

    gko::mpi::init_finalize mpi_init{argc, argv, 1};
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = gko::MpiExecutor::create(
        exec_map.at(executor_string)());  // throws if not valid
    const auto sub_exec = exec->get_sub_executor();
    auto num_ranks = exec->get_num_ranks(exec->get_communicator());
    auto my_rank = exec->get_my_rank(exec->get_communicator());
    auto root_rank = exec->get_root_rank();

    // Naive row distribution
    gko::size_type size = 19;
    auto local_num_rows = static_cast<gko::size_type>(size / num_ranks);
    auto dist = gko::Array<gko::size_type>{
        sub_exec, (my_rank == num_ranks - 1) ? size - my_rank * local_num_rows
                                             : local_num_rows};
    std::iota(dist.get_data(), dist.get_data() + dist.get_num_elems(),
              my_rank * local_num_rows);

    // Create distributed matrix
    auto A =
        share(gko::read_dist<mtx>(std::ifstream("data/A.mtx"), dist, exec));
    auto b = gko::read_dist<vec>(std::ifstream("data/b.mtx"), dist, exec);
    auto x = gko::read_dist<vec>(std::ifstream("data/x0.mtx"), dist, exec);

    const RealValueType reduction_factor{1e-7};
    auto solver_gen =
        cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(20u).on(sub_exec),
                gko::stop::ResidualNormReduction<ValueType>::build()
                    .with_reduction_factor(reduction_factor)
                    .on(sub_exec))
            .on(exec);
    auto solver = solver_gen->generate(A);
    solver->apply(lend(b), lend(x));

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    std::cout << " Rank: " << my_rank << " with A local size " << A->get_size()
              << std::endl;
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(res));

    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, lend(res));
}
