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

#ifndef GKO_INCLUDE_CONFIG_H
#define GKO_INCLUDE_CONFIG_H

// clang-format off
#define GKO_VERSION_MAJOR 1
#define GKO_VERSION_MINOR 3
#define GKO_VERSION_PATCH 0
#define GKO_VERSION_TAG "develop"
#define GKO_VERSION_STR 1, 3, 0
// clang-format on

/*
 * Controls the amount of messages output by Ginkgo.
 * 0 disables all output (except for test, benchmarks and examples).
 * 1 activates important messages.
 */
// clang-format off
#define GKO_VERBOSE_LEVEL 1
// clang-format on


/* Is Itanium ABI available? */
#define GKO_HAVE_CXXABI_H


/* Should we use all optimizations for Jacobi? */
/* #undef GINKGO_JACOBI_FULL_OPTIMIZATIONS */


/* What is HIP compiled for, hcc or nvcc? */
// clang-format off
#define GINKGO_HIP_PLATFORM_HCC 0


#define GINKGO_HIP_PLATFORM_NVCC 0
// clang-format on


/* Is PAPI SDE available for Logging? */
// clang-format off
#define GKO_HAVE_PAPI_SDE 0
// clang-format on


#endif  // GKO_INCLUDE_CONFIG_H