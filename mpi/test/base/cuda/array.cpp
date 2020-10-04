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
class DistributedArray : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    DistributedArray() : mpi_exec(nullptr) {}

    void SetUp()
    {
        char **argv;
        int argc = 0;
        exec = gko::ReferenceExecutor::create();
        host = gko::ReferenceExecutor::create();
        mpi_exec2 = gko::MpiExecutor::create(host);
        auto comm2 = mpi_exec2->get_communicator();
        rank = mpi_exec2->get_my_rank(comm2);
        cuda_exec = gko::CudaExecutor::create(rank, host);
        mpi_exec = gko::MpiExecutor::create(cuda_exec);
        auto comm = mpi_exec->get_communicator();
        sub_exec = mpi_exec->get_sub_executor();
        auto ndev = gko::as<gko::CudaExecutor>(mpi_exec->get_sub_executor())
                        ->get_num_devices();
        rank = mpi_exec->get_my_rank(comm);
        GKO_ASSERT(ndev > 0);
        auto dev_id =
            gko::as<gko::CudaExecutor>(sub_exec.get())->get_device_id();
        auto dev_id_mspace =
            gko::as<gko::CudaMemorySpace>(
                gko::as<gko::CudaExecutor>(sub_exec.get())->get_mem_space())

                ->get_device_id();
        ASSERT_GT(mpi_exec->get_num_ranks(comm), 1);
    }

    void TearDown()
    {
        if (mpi_exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(mpi_exec->synchronize());
        }
    }

    static void assert_empty(gko::matrix::Dense<value_type> *m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
    }

    static void assert_equal_arrays(const gko::Array<value_type> &m,
                                    const gko::Array<value_type> &lm)
    {
        ASSERT_EQ(m.get_num_elems(), lm.get_num_elems());

        for (auto i = 0; i < m.get_num_elems(); ++i) {
            EXPECT_EQ(m.get_const_data()[i], lm.get_const_data()[i]);
        }
    }

    std::shared_ptr<gko::MpiExecutor> mpi_exec;
    std::shared_ptr<gko::MpiExecutor> mpi_exec2;
    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<gko::Executor> host;
    std::shared_ptr<gko::Executor> cuda_exec;
    std::shared_ptr<const gko::Executor> sub_exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    int rank;
};

TYPED_TEST_CASE(DistributedArray, gko::test::ValueTypes);


TYPED_TEST(DistributedArray, CanDistributeArray)
{
    using value_type = typename TestFixture::value_type;
    value_type *data;
    value_type *comp_data;
    auto sub_exec = this->mpi_exec->get_sub_executor();
    gko::Array<value_type> m{this->mpi_exec};
    gko::Array<value_type> mh{this->host};
    gko::Array<value_type> orig_array{sub_exec};
    gko::Array<value_type> lm{sub_exec->get_master()};
    gko::IndexSet<gko::int32> index_set{20};
    this->mpi_exec->set_root_rank(0);
    if (this->rank == 0) {
        // clang-format off
        data = new value_type[20]{
                                 1.0, 2.0, -1.0, 2.0,
                                     0.0, 4.0, -2.0, 9.0,
                                     3.0, 7.0, 1.0, 0.2,
                                     -3.0, 4.0, -1.0, 2.0,
                                     5.0, 6.0, -1.0, 4.0};
        comp_data = new value_type[8]{
                                      1.0, 2.0, -1.0, 2.0,
                                          0.0, 4.0, -2.0, 9.0};
        // clang-format on
        orig_array = gko::Array<value_type>(
            sub_exec,
            gko::Array<value_type>::view(sub_exec->get_master(), 20, data));
        lm = gko::Array<value_type>(
            sub_exec->get_master(),
            gko::Array<value_type>::view(sub_exec->get_master(), 8, comp_data));
        index_set.add_subset(0, 8);
    } else {
        // clang-format off
        comp_data = new value_type[12]{3.0, 7.0, 1.0, 0.2,
                -3.0, 4.0, -1.0, 2.0,
                5.0, 6.0, -1.0, 4.0};
        // clang-format on
        index_set.add_subset(8, 20);
        lm = gko::Array<value_type>(sub_exec->get_master(),
                                    gko::Array<value_type>::view(
                                        sub_exec->get_master(), 12, comp_data));
    }
    m = orig_array.distribute(this->mpi_exec, index_set);
    mh = m;
    ASSERT_EQ(m.get_executor(), this->mpi_exec);
    this->assert_equal_arrays(mh, lm);
    if (this->rank == 0) {
        delete data;
    }
    delete comp_data;
}


TYPED_TEST(DistributedArray, CanDistributeNonContiguousArrays)
{
    using value_type = typename TestFixture::value_type;
    value_type *data;
    value_type *comp_data;
    auto sub_exec = this->mpi_exec->get_sub_executor();
    gko::Array<value_type> m{this->mpi_exec};
    gko::Array<value_type> mh{this->host};
    gko::Array<value_type> orig_array{sub_exec->get_master()};
    gko::Array<value_type> lm{sub_exec->get_master()};
    gko::IndexSet<gko::int32> index_set{20};
    this->mpi_exec->set_root_rank(0);
    if (this->rank == 0) {
        data = new value_type[20]{1.0, 2.0,  -1.0, 2.0,  5.0,  4.0, 1.0,
                                  7.0, 2.0,  -2.0, -1.0, 3.0,  3.0, 8.0,
                                  1.0, -3.0, 5.0,  6.0,  -1.0, 4.0};
        comp_data = new value_type[10]{1.0, 2.0, -1.0, 2.0,  5.0,
                                       3.0, 8.0, 1.0,  -3.0, 5.0};
        orig_array = gko::Array<value_type>(
            sub_exec->get_master(),
            gko::Array<value_type>::view(sub_exec->get_master(), 20, data));
        lm = gko::Array<TypeParam>(sub_exec->get_master(),
                                   gko::Array<value_type>::view(
                                       sub_exec->get_master(), 10, comp_data));
        index_set.add_subset(0, 5);
        index_set.add_subset(12, 17);
    } else {
        comp_data = new value_type[10]{4.0,  1.0, 7.0, 2.0,  -2.0,
                                       -1.0, 3.0, 6.0, -1.0, 4.0};
        index_set.add_subset(5, 12);
        index_set.add_subset(17, 20);
        lm = gko::Array<TypeParam>(sub_exec->get_master(),
                                   gko::Array<value_type>::view(
                                       sub_exec->get_master(), 10, comp_data));
    }
    m = orig_array.distribute(this->mpi_exec, index_set);
    mh = m;
    ASSERT_EQ(m.get_executor(), this->mpi_exec);
    this->assert_equal_arrays(mh, lm);
    if (this->rank == 0) {
        delete data;
    }
    delete comp_data;
}


TYPED_TEST(DistributedArray, CanGatherArrayOnRoot)
{
    using value_type = typename TestFixture::value_type;
    value_type *data;
    value_type *comp_data;
    auto sub_exec = this->mpi_exec->get_sub_executor();
    gko::Array<value_type> coll_array{this->mpi_exec};
    gko::Array<value_type> coll_array_h{this->mpi_exec2};
    gko::Array<value_type> comp_array{this->sub_exec->get_master()};
    gko::Array<value_type> local_array{sub_exec};
    gko::IndexSet<gko::int32> index_set{20};
    this->mpi_exec->set_root_rank(0);
    if (this->rank == 0) {
        // clang-format off
        comp_data = new value_type[20]{
                                      1.0, 2.0, -1.0, 2.0,
                                          0.0, 4.0, -2.0, 9.0,
                                          3.0, 7.0, 1.0, 0.2,
                                          -3.0, 4.0, -1.0, 2.0,
                                          5.0, 6.0, -1.0, 4.0};
        data = new value_type[8]{
                                 1.0, 2.0, -1.0, 2.0,
                                     0.0, 4.0, -2.0, 9.0};
        // clang-format on
        comp_array = gko::Array<value_type>(
            sub_exec->get_master(), gko::Array<value_type>::view(
                                        sub_exec->get_master(), 20, comp_data));
        local_array = gko::Array<value_type>(
            sub_exec,
            gko::Array<value_type>::view(sub_exec->get_master(), 8, data));
        index_set.add_subset(0, 8);
    } else {
        // clang-format off
        data = new value_type[12]{3.0, 7.0, 1.0, 0.2,
                                       -3.0, 4.0, -1.0, 2.0,
                                       5.0, 6.0, -1.0, 4.0};
        // clang-format on
        index_set.add_subset(8, 20);
        local_array = gko::Array<value_type>(
            sub_exec,
            gko::Array<value_type>::view(sub_exec->get_master(), 12, data));
    }
    coll_array = local_array.gather_on_root(this->mpi_exec, index_set);
    coll_array_h = coll_array;
    if (this->rank == 0) {
        ASSERT_EQ(coll_array.get_executor(), this->mpi_exec);
        this->assert_equal_arrays(comp_array, coll_array_h);
        delete comp_data;
    }
    delete data;
}


TYPED_TEST(DistributedArray, CanGatherNonContiguousArraysOnRoot)
{
    using value_type = typename TestFixture::value_type;
    value_type *data;
    value_type *comp_data;
    auto sub_exec = this->mpi_exec->get_sub_executor();
    gko::Array<value_type> coll_array{this->mpi_exec};
    gko::Array<value_type> coll_array_h{this->mpi_exec2};
    gko::Array<value_type> comp_array{sub_exec->get_master()};
    gko::Array<value_type> local_array{sub_exec};
    gko::IndexSet<gko::int32> index_set{20};
    this->mpi_exec->set_root_rank(0);
    if (this->rank == 0) {
        comp_data = new value_type[20]{1.0, 2.0,  -1.0, 2.0,  5.0,  4.0, 1.0,
                                       7.0, 2.0,  -2.0, -1.0, 3.0,  3.0, 8.0,
                                       1.0, -3.0, 5.0,  6.0,  -1.0, 4.0};
        data = new value_type[10]{1.0, 2.0, -1.0, 2.0,  5.0,
                                  3.0, 8.0, 1.0,  -3.0, 5.0};
        comp_array = gko::Array<value_type>(
            sub_exec->get_master(), gko::Array<value_type>::view(
                                        sub_exec->get_master(), 20, comp_data));
        local_array = gko::Array<TypeParam>(
            sub_exec,
            gko::Array<value_type>::view(sub_exec->get_master(), 10, data));
        index_set.add_subset(0, 5);
        index_set.add_subset(12, 17);
    } else {
        data = new value_type[10]{4.0,  1.0, 7.0, 2.0,  -2.0,
                                  -1.0, 3.0, 6.0, -1.0, 4.0};
        index_set.add_subset(5, 12);
        index_set.add_subset(17, 20);
        local_array = gko::Array<TypeParam>(
            sub_exec,
            gko::Array<value_type>::view(sub_exec->get_master(), 10, data));
    }
    coll_array = local_array.gather_on_root(this->mpi_exec, index_set);
    coll_array_h = coll_array;
    if (this->rank == 0) {
        ASSERT_EQ(coll_array.get_executor(), this->mpi_exec);
        this->assert_equal_arrays(coll_array_h, comp_array);
        delete comp_data;
    }
    delete data;
}


TYPED_TEST(DistributedArray, CanGatherNonContiguousArraysOnAllRanks)
{
    using value_type = typename TestFixture::value_type;
    value_type *data;
    value_type *comp_data;
    auto sub_exec = this->mpi_exec->get_sub_executor();
    gko::Array<value_type> coll_array{this->mpi_exec};
    gko::Array<value_type> coll_array_h{this->mpi_exec2};
    gko::Array<value_type> comp_array{sub_exec->get_master()};
    gko::Array<value_type> local_array{sub_exec};
    gko::IndexSet<gko::int32> index_set{20};
    this->mpi_exec->set_root_rank(0);
    if (this->rank == 0) {
        data = new value_type[10]{1.0, 2.0, -1.0, 2.0,  5.0,
                                  3.0, 8.0, 1.0,  -3.0, 5.0};
        local_array = gko::Array<TypeParam>(
            sub_exec,
            gko::Array<value_type>::view(sub_exec->get_master(), 10, data));
        index_set.add_subset(0, 5);
        index_set.add_subset(12, 17);
    } else {
        data = new value_type[10]{4.0,  1.0, 7.0, 2.0,  -2.0,
                                  -1.0, 3.0, 6.0, -1.0, 4.0};
        index_set.add_subset(5, 12);
        index_set.add_subset(17, 20);
        local_array = gko::Array<TypeParam>(
            sub_exec,
            gko::Array<value_type>::view(sub_exec->get_master(), 10, data));
    }
    comp_data = new value_type[20]{1.0, 2.0,  -1.0, 2.0,  5.0,  4.0, 1.0,
                                   7.0, 2.0,  -2.0, -1.0, 3.0,  3.0, 8.0,
                                   1.0, -3.0, 5.0,  6.0,  -1.0, 4.0};
    comp_array = gko::Array<value_type>(
        sub_exec->get_master(),
        gko::Array<value_type>::view(sub_exec->get_master(), 20, comp_data));

    coll_array = local_array.gather_on_all(this->mpi_exec, index_set);
    coll_array_h = coll_array;

    ASSERT_EQ(coll_array.get_executor(), this->mpi_exec);
    this->assert_equal_arrays(coll_array_h, comp_array);
    delete comp_data;
    delete data;
}


}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
