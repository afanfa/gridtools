/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;

namespace rw_test {

#ifdef __CUDACC__
    using backend_t = ::gridtools::backend< Cuda, GRIDBACKEND, Block >;
#else
#ifdef BACKEND_BLOCK
    using backend_t = ::gridtools::backend< Host, GRIDBACKEND, Block >;
#else
    using backend_t = ::gridtools::backend< Host, GRIDBACKEND, Naive >;
#endif
#endif

    using icosahedral_topology_t = ::gridtools::icosahedral_topology< backend_t >;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    struct test_functor {
        typedef accessor< 0, in, icosahedral_topology_t::cells, extent< 0 > > i0;
        typedef accessor< 1, inout, icosahedral_topology_t::cells > o0;
        typedef accessor< 2, in, icosahedral_topology_t::cells, extent< 2 > > i1;
        typedef accessor< 3, inout, icosahedral_topology_t::cells > o1;
        typedef accessor< 4, in, icosahedral_topology_t::cells, extent< 3 > > i2;
        typedef accessor< 5, inout, icosahedral_topology_t::cells > o2;
        typedef accessor< 6, in, icosahedral_topology_t::cells, extent< 4 > > i3;
        typedef accessor< 7, inout, icosahedral_topology_t::cells > o3;

        typedef boost::mpl::vector8< i0, o0, i1, o1, i2, o2, i3, o3 > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {}
    };
} // namespace rw_test

using namespace rw_test;

TEST(esf, read_write) {

    typedef gridtools::layout_map< 2, 1, 0 > layout_t;

    using cell_storage_type = typename backend_t::storage_t< icosahedral_topology_t::cells, double >;

    typedef arg< 0, cell_storage_type > pi0;
    typedef arg< 1, cell_storage_type > po0;
    typedef arg< 2, cell_storage_type > pi1;
    typedef arg< 3, cell_storage_type > po1;
    typedef arg< 4, cell_storage_type > pi2;
    typedef arg< 5, cell_storage_type > po2;
    typedef arg< 6, cell_storage_type > pi3;
    typedef arg< 7, cell_storage_type > po3;

    typedef boost::mpl::vector8< pi0, po0, pi1, po1, pi2, po2, pi3, po3 > args;

    typedef esf_descriptor< test_functor, icosahedral_topology_t, icosahedral_topology_t::cells, args > esf_t;

    GRIDTOOLS_STATIC_ASSERT(is_accessor_readonly< test_functor::i0 >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(!is_accessor_readonly< test_functor::o0 >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(is_accessor_readonly< test_functor::i1 >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(!is_accessor_readonly< test_functor::o1 >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(is_accessor_readonly< test_functor::i2 >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(!is_accessor_readonly< test_functor::o2 >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(is_accessor_readonly< test_functor::i3 >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(!is_accessor_readonly< test_functor::o3 >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(!is_written< esf_t >::apply< static_int< 0 > >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(is_written< esf_t >::apply< static_int< 1 > >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(!is_written< esf_t >::apply< static_int< 2 > >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(is_written< esf_t >::apply< static_int< 3 > >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(!is_written< esf_t >::apply< static_int< 4 > >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(is_written< esf_t >::apply< static_int< 5 > >::type::value, "");

    GRIDTOOLS_STATIC_ASSERT(!is_written< esf_t >::apply< static_int< 6 > >::type::value, "");
    GRIDTOOLS_STATIC_ASSERT(is_written< esf_t >::apply< static_int< 7 > >::type::value, "");

    EXPECT_TRUE(true);
}