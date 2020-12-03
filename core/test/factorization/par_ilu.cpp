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

#include <ginkgo/core/factorization/par_ilu.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


namespace {


class ParIlu : public ::testing::Test {
public:
    using value_type = double;
    using index_type = gko::int32;
    using ilu_factory_type = gko::factorization::ParIlu<value_type, index_type>;
    using strategy_type = ilu_factory_type::matrix_type::classical;

protected:
    ParIlu() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};


TEST_F(ParIlu, SetIterations)
{
    auto factory = ilu_factory_type::build().with_iterations(5u).on(this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 5u);
}


TEST_F(ParIlu, SetSkip)
{
    auto factory =
        ilu_factory_type::build().with_skip_sorting(true).on(this->ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, true);
}


TEST_F(ParIlu, SetLStrategy)
{
    auto strategy = std::make_shared<strategy_type>();

    auto factory =
        ilu_factory_type::build().with_l_strategy(strategy).on(this->ref);

    ASSERT_EQ(factory->get_parameters().l_strategy, strategy);
}


TEST_F(ParIlu, SetUStrategy)
{
    auto strategy = std::make_shared<strategy_type>();

    auto factory =
        ilu_factory_type::build().with_u_strategy(strategy).on(this->ref);

    ASSERT_EQ(factory->get_parameters().u_strategy, strategy);
}


TEST_F(ParIlu, SetDefaults)
{
    auto factory = ilu_factory_type::build().on(this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 0u);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().l_strategy, nullptr);
    ASSERT_EQ(factory->get_parameters().u_strategy, nullptr);
}


TEST_F(ParIlu, SetEverything)
{
    auto strategy = std::make_shared<strategy_type>();
    auto strategy2 = std::make_shared<strategy_type>();

    auto factory = ilu_factory_type::build()
                       .with_iterations(7u)
                       .with_skip_sorting(false)
                       .with_l_strategy(strategy)
                       .with_u_strategy(strategy2)
                       .on(this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 7u);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().l_strategy, strategy);
    ASSERT_EQ(factory->get_parameters().u_strategy, strategy2);
}


}  // namespace
