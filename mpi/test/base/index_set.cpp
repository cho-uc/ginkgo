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


#include <mpi.h>

#include <gtest/gtest.h>

#include "gtest-mpi-listener.hpp"
#include "gtest-mpi-main.hpp"

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class IndexSet : public ::testing::Test {
protected:
    using value_type = T;
    IndexSet() : exec(nullptr) {}

    void SetUp()
    {
        char **argv;
        int argc = 0;
        exec = gko::MpiExecutor::create(gko::ReferenceExecutor::create());
        sub_exec = exec->get_sub_executor();
        auto comm = exec->get_communicator();
        num_ranks = exec->get_num_ranks(comm);
        rank = exec->get_my_rank(comm);
        ASSERT_GT(num_ranks, 1);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    static void assert_equal_to_original(gko::IndexSet<T> &a)
    {
        ASSERT_EQ(a.get_size(), 10);
    }


    std::shared_ptr<gko::MpiExecutor> exec;
    std::shared_ptr<const gko::Executor> sub_exec;
    int rank;
    int num_ranks;
};

TYPED_TEST_CASE(IndexSet, gko::test::IndexTypes);


TYPED_TEST(IndexSet, CanCheckWhenAscendingAndOneToOne)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 12};

    if (this->rank == 0) {
        idx_set.add_subset(0, 3);
    } else if (this->rank == 1) {
        idx_set.add_subset(3, 6);
    } else if (this->rank == 2) {
        idx_set.add_subset(6, 9);
    } else if (this->rank == 3) {
        idx_set.add_subset(9, 12);
    }

    ASSERT_TRUE(idx_set.is_ascending_and_one_to_one());
}


TYPED_TEST(IndexSet, CanCheckWhenNotAscendingAndOneToOne)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 12};

    if (this->rank == 0) {
        idx_set.add_subset(0, 3);
    } else if (this->rank == 1) {
        idx_set.add_subset(3, 6);
    } else if (this->rank == 2) {
        idx_set.add_subset(6, 8);
    } else if (this->rank == 3) {
        idx_set.add_subset(9, 12);
    }

    ASSERT_FALSE(idx_set.is_ascending_and_one_to_one());
}


}  // namespace


// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
