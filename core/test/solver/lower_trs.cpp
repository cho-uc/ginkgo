/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/solver/lower_trs.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/cg.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class LowerTrs : public ::testing::Test {
protected:
    using Solver = gko::solver::LowerTrs<>;
    using CgSolver = gko::solver::Cg<>;

    LowerTrs()
        : exec(gko::ReferenceExecutor::create()),
          prec_fac(CgSolver::build().on(exec)),
          lower_trs_factory(
              Solver::build().with_preconditioner(prec_fac).on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<CgSolver::Factory> prec_fac;
    std::unique_ptr<Solver::Factory> lower_trs_factory;
};


TEST_F(LowerTrs, LowerTrsFactoryKnowsItsExecutor)
{
    ASSERT_EQ(lower_trs_factory->get_executor(), exec);
}


TEST_F(LowerTrs, LowerTrsFactoryKnowsItsPrecond)
{
    ASSERT_EQ(static_cast<const CgSolver::Factory *>(
                  lower_trs_factory->get_parameters().preconditioner.get()),
              prec_fac.get());
}


}  // namespace