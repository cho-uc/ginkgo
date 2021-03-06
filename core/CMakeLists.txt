add_subdirectory(devices)  # basic device functionalities, always compiled
add_subdirectory(device_hooks)  # placeholders for disabled modules

add_library(ginkgo "")
target_sources(ginkgo
    PRIVATE
    base/array.cpp
    base/combination.cpp
    base/composition.cpp
    base/executor.cpp
    base/mtx_io.cpp
    base/perturbation.cpp
    base/version.cpp
    factorization/ilu.cpp
    factorization/par_ict.cpp
    factorization/par_ilu.cpp
    factorization/par_ilut.cpp
    log/convergence.cpp
    log/logger.cpp
    log/record.cpp
    log/stream.cpp
    matrix/coo.cpp
    matrix/csr.cpp
    matrix/dense.cpp
    matrix/diagonal.cpp
    matrix/ell.cpp
    matrix/hybrid.cpp
    matrix/identity.cpp
    matrix/permutation.cpp
    matrix/sellp.cpp
    matrix/sparsity_csr.cpp
    preconditioner/isai.cpp
    preconditioner/jacobi.cpp
    reorder/rcm.cpp
    solver/bicg.cpp
    solver/bicgstab.cpp
    solver/cg.cpp
    solver/cgs.cpp
    solver/fcg.cpp
    solver/gmres.cpp
    solver/idr.cpp
    solver/ir.cpp
    solver/lower_trs.cpp
    solver/upper_trs.cpp
    stop/combined.cpp
    stop/criterion.cpp
    stop/iteration.cpp
    stop/residual_norm.cpp
    stop/time.cpp)

if(GINKGO_HAVE_PAPI_SDE)
    target_sources(ginkgo PRIVATE log/papi.cpp)
endif()

ginkgo_compile_features(ginkgo)

target_compile_options(ginkgo PRIVATE "${GINKGO_COMPILER_FLAGS}")
# add a namespace alias so Ginkgo can always be included as Ginkgo::ginkgo
# regardless of whether it is installed or added as a subdirectory
add_library(Ginkgo::ginkgo ALIAS ginkgo)
target_link_libraries(ginkgo
    PUBLIC ginkgo_omp ginkgo_cuda ginkgo_reference ginkgo_hip ginkgo_dpcpp)
# The PAPI dependency needs to be exposed to the user.
if (GINKGO_HAVE_PAPI_SDE)
    target_link_libraries(ginkgo PUBLIC PAPI::PAPI)
endif()
ginkgo_default_includes(ginkgo)
ginkgo_install_library(ginkgo core)

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo)
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
