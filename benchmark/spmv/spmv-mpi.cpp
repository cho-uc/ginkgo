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

#include <ginkgo/ginkgo.hpp>


#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <typeinfo>


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/spmv_common.hpp"


using etype = double;


// Command-line arguments

DEFINE_uint32(nrhs, 1, "The number of right hand sides");

DEFINE_bool(distributed_rhs, false, "If true, uses a distributed rhs");

// This function supposes that management of `FLAGS_overwrite` is done before
// calling it
void apply_spmv(const char *format_name, const char *row_dist_type,
                std::shared_ptr<gko::Executor> exec,
                const gko::matrix_data<etype> &data,
                const gko::Array<gko::size_type> &row_dist, const vec<etype> *b,
                const vec<etype> *x, const vec<etype> *answer,
                rapidjson::Value &test_case,
                rapidjson::MemoryPoolAllocator<> &allocator)
{
    try {
        auto mpi_exec = gko::as<gko::MpiExecutor>(exec.get());
        auto sub_exec = exec->get_sub_executor();
        auto rank = mpi_exec->get_my_rank();
        auto &spmv_case = test_case["spmv"];
        auto &format_case = spmv_case[format_name];
        add_or_set_member(spmv_case[format_name], row_dist_type,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto storage_logger = std::make_shared<StorageLogger>(exec);
        exec->add_logger(storage_logger);
        auto tic = std::chrono::steady_clock::now();
        auto system_matrix = share(
            formats::matrix_factory_dist.at(format_name)(exec, data, row_dist));
        auto toc = std::chrono::steady_clock::now();

        add_or_set_member(
            format_case[row_dist_type], "mat-distribute",
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic)
                    .count()),
            allocator);
        exec->remove_logger(gko::lend(storage_logger));
        storage_logger->write_data(format_case[row_dist_type], allocator);
        // check the residual
        if (FLAGS_detailed) {
            auto x_clone = clone(x);
            exec->synchronize();
            system_matrix->apply(lend(b), lend(x_clone));
            exec->synchronize();
            double max_relative_norm2 =
                compute_max_relative_norm2(lend(x_clone), lend(answer));
            add_or_set_member(format_case[format_name], "max_relative_norm2",
                              max_relative_norm2, allocator);
        }
        // warm run
        for (unsigned int i = 0; i < FLAGS_warmup; i++) {
            auto x_clone = clone(x);
            exec->synchronize();
            system_matrix->apply(lend(b), lend(x_clone));
            exec->synchronize();
        }
        std::chrono::nanoseconds time(0);
        // timed run
        for (unsigned int i = 0; i < FLAGS_repetitions; i++) {
            auto x_clone = clone(x);
            exec->synchronize();
            tic = std::chrono::steady_clock::now();
            system_matrix->apply(lend(b), lend(x_clone));

            exec->synchronize();
            toc = std::chrono::steady_clock::now();
            time +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
        }
        add_or_set_member(format_case[row_dist_type], "time",
                          static_cast<double>(time.count()) / FLAGS_repetitions,
                          allocator);

        // compute and write benchmark data
        add_or_set_member(format_case[row_dist_type], "completed", true,
                          allocator);
    } catch (const std::exception &e) {
        add_or_set_member(test_case["spmv"][format_name][row_dist_type],
                          "completed", false, allocator);
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char *argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's spmv.\n";
    std::string format = std::string() + "  [\n" +
                         "    { \"filename\": \"my_file.mtx\"},\n" +
                         "    { \"filename\": \"my_file2.mtx\"}\n" + "  ]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    std::string extra_information = "The formats are " + FLAGS_formats +
                                    "\nThe number of right hand sides is " +
                                    std::to_string(FLAGS_nrhs) + "\n";
    print_general_information(extra_information);

    gko::mpi::init_finalize mpi_init{argc, argv, 1};
    auto exec = executor_factory.at(FLAGS_executor)();
    auto mpi_exec = gko::as<const gko::MpiExecutor>(exec.get());
    auto host_mpi_exec = gko::MpiExecutor::create(exec->get_master());
    auto rank = mpi_exec->get_my_rank();
    auto num_ranks = mpi_exec->get_num_ranks();

    auto engine = get_engine();
    auto formats = split(FLAGS_formats, ',');
    auto row_dists_vec = split(FLAGS_row_dists, ',');

    std::ifstream ifs(FLAGS_filename);
    rapidjson::IStreamWrapper jc_ifs(ifs);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jc_ifs);


    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto &allocator = test_cases.GetAllocator();

    for (auto &test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("spmv")) {
                test_case.AddMember("spmv",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto &spmv_case = test_case["spmv"];
            if (!FLAGS_overwrite &&
                all_of(begin(formats), end(formats),
                       [&spmv_case](const std::string &s) {
                           return spmv_case.HasMember(s.c_str());
                       }) &&
                all_of(begin(row_dists_vec), end(row_dists_vec),
                       [&spmv_case](const std::string &s) {
                           return spmv_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;
            std::ifstream mtx_fd(test_case["filename"].GetString());
            auto data = gko::read_raw<etype>(mtx_fd);
            auto row_d = distributed::row_distribution.at("equal")(exec, data);

            auto nrhs = FLAGS_nrhs;
            auto b = vec<etype>::create(exec);
            auto x = vec<etype>::create(exec);
            if (FLAGS_distributed_rhs) {
                b = create_matrix<etype>(exec, gko::dim<2>{data.size[1], nrhs},
                                         row_d, engine);
                x = create_matrix<etype>(exec, gko::dim<2>{data.size[0], nrhs},
                                         row_d, engine);
            } else {
                b = create_matrix<etype>(exec, gko::dim<2>{data.size[1], nrhs},
                                         engine);
                x = create_matrix<etype>(exec, gko::dim<2>{data.size[0], nrhs},
                                         engine);
            }
            std::string best_format_and_dist("none");
            auto best_performance = 0.0;
            if (!test_case.HasMember("optimal")) {
                test_case.AddMember("optimal",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }

            // Compute the result from ginkgo::coo as the correct answer
            auto answer = vec<etype>::create(exec);
            if (FLAGS_detailed) {
                auto row_dist =
                    distributed::row_distribution.at("equal")(exec, data);
                if (FLAGS_debug) {
                    GKO_ASSERT_CONDITION(
                        distributed::verify_dist(exec, host_mpi_exec, row_dist,
                                                 data.size[0]) == true);
                }
                auto system_matrix = share(formats::matrix_factory_dist.at(
                    "coo")(exec, data, row_dist));
                std::clog << "Matrix is of size " << data.size
                          << " Distributed matrix size "
                          << system_matrix->get_size()
                          << " Distributed rhs size " << b->get_size()
                          << std::endl;
                answer->copy_from(lend(x));
                exec->synchronize();
                system_matrix->apply(lend(b), lend(answer));
                exec->synchronize();
            }
            for (const auto &format_name : formats) {
                auto &spmv_case = test_case["spmv"];
                add_or_set_member(spmv_case, format_name.c_str(),
                                  rapidjson::Value(rapidjson::kObjectType),
                                  allocator);
                for (const auto &row_dist_type : row_dists_vec) {
                    auto row_dist = distributed::row_distribution.at(
                        row_dist_type)(exec, data);
                    if (FLAGS_debug) {
                        GKO_ASSERT_CONDITION(distributed::verify_dist(
                            exec, host_mpi_exec, row_dist, data.size[0]));
                    }
                    exec->synchronize();
                    apply_spmv(format_name.c_str(), row_dist_type.c_str(), exec,
                               data, row_dist, lend(b), lend(x), lend(answer),
                               test_case, allocator);
                    exec->synchronize();
                    std::clog << "Current state:" << std::endl
                              << test_cases << std::endl;
                    exec->synchronize();
                    if (spmv_case[format_name.c_str()][row_dist_type.c_str()]
                                 ["completed"]
                                     .GetBool()) {
                        auto performance =
                            spmv_case[format_name.c_str()]
                                     [row_dist_type.c_str()]["time"]
                                         .GetDouble();
                        if (best_format_and_dist == "none" ||
                            performance < best_performance) {
                            best_format_and_dist =
                                format_name + "-" + row_dist_type;
                            best_performance = performance;
                            add_or_set_member(
                                test_case["optimal"], "spmv",
                                rapidjson::Value(best_format_and_dist.c_str(),
                                                 allocator)
                                    .Move(),
                                allocator);
                        }
                    }
                    backup_results(test_cases);
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
        }
    }

    std::string fname = FLAGS_filename;
    auto fsubstr = fname.substr(0, fname.size() - 9);
    std::ofstream ofs(fsubstr + "_" + std::to_string(rank) + ".json");
    rapidjson::OStreamWrapper jc_ofs(ofs);
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper, rapidjson::UTF8<>,
                            rapidjson::UTF8<>, rapidjson::CrtAllocator,
                            rapidjson::kWriteNanAndInfFlag>
        writer(jc_ofs);
    test_cases.Accept(writer);

    std::cout << test_cases << std::endl;
}
