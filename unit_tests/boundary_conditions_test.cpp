#include <gridtools.h>
#include <common/halo_descriptor.h>

#ifdef CUDA_EXAMPLE
#include <boundary-conditions/apply_gpu.h>
#else
#include <boundary-conditions/apply.h>
#endif

using gridtools::direction;
using gridtools::sign;
using gridtools::minus;
using gridtools::zero;
using gridtools::plus;

#ifdef CUDA_EXAMPLE
#include <stencil-composition/backend_cuda.h>
#else
#include <stencil-composition/backend_block.h>
#include <stencil-composition/backend_naive.h>
#endif

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>

#include <stdlib.h>
#include <stdio.h>

#ifdef CUDA_EXAMPLE
#define BACKEND backend_cuda
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend_block
#else
#define BACKEND backend_naive
#endif
#endif


struct bc_basic {

    // relative coordinates
    template <typename Direction, typename DataField0>
    GT_FUNCTION
    void operator()(Direction,
                    DataField0 & data_field0,
                    int i, int j, int k) const {
        data_field0(i,j,k) = i+j+k;
    }
};

struct minus_predicate {
    template <typename Direction>
    bool operator()(Direction) const {
        return true;
    }

    template <sign I, sign J>
    bool operator()(direction<I,J,minus>) const {
        return false;
    }

    template <sign I, sign K>
    bool operator()(direction<I,minus,K>) const {
        return false;
    }

    template <sign J, sign K>
    bool operator()(direction<minus,J,K>) const {
        return false;
    }


    template <sign I>
    bool operator()(direction<I,minus,minus>) const {
        return false;
    }

    template <sign I>
    bool operator()(direction<minus,I,minus>) const {
        return false;
    }

    template <sign I>
    bool operator()(direction<minus,minus,I>) const {
        return false;
    }

    bool operator()(direction<minus,minus,minus>) const {
        return false;
    }

};

bool basic() {

    int d1 = 5;
    int d2 = 5;
    int d3 = 5;

    typedef gridtools::BACKEND::storage_type<int, gridtools::layout_map<0,1,2> >::type storage_type;

    // Definition of the actual data fields that are used for input/output
    storage_type in(d1,d2,d3,-1, std::string("in"));
    storage_type out(d1,d2,d3,-7.3, std::string("out"));
    storage_type coeff(d1,d2,d3,8, std::string("coeff"));

    for (int i=0; i<d1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<d3; ++k) {
                in(i,j,k) = 0;
                out(i,j,k) = 0;
            }
        }
    }

#ifndef NDEBUG
    for (int i=0; i<d1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
    halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
    halos[2] = gridtools::halo_descriptor(1,1,1,d3-2,d3);

#ifdef CUDA_EXAMPLE
    in.clone_to_gpu();
    out.clone_to_gpu();
    in.h2d_update();
    out.h2d_update();

    gridtools::boundary_apply_gpu<bc_basic>(halos, bc_basic()).apply(in);

    in.d2h_update();
#else
    gridtools::boundary_apply<bc_basic>(halos, bc_basic()).apply(in);
#endif

#ifndef NDEBUG
    for (int i=0; i<d1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    bool result = true;

    for (int i=0; i<d1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<1; ++k) {
                if (in(i,j,k) != i+j+k) {
                    result = false;
                }
            }
        }
    }

    for (int i=0; i<d1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=d3-1; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    result = false;
                }
            }
        }
    }

    for (int i=0; i<d1; ++i) {
        for (int j=0; j<1; ++j) {
            for (int k=0; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    result = false;
                }
            }
        }
    }

    for (int i=0; i<d1; ++i) {
        for (int j=d2-1; j<d2; ++j) {
            for (int k=0; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    result = false;
                }
            }
        }
    }

    for (int i=0; i<1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    result = false;
                }
            }
        }
    }

    for (int i=d1-1; i<d1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    result = false;
                }
            }
        }
    }

    for (int i=1; i<d1-1; ++i) {
        for (int j=1; j<d2-1; ++j) {
            for (int k=1; k<d3-1; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }
    
    return result;

}

bool predicate() {

    int d1 = 5;
    int d2 = 5;
    int d3 = 5;

    typedef gridtools::BACKEND::storage_type<int, gridtools::layout_map<0,1,2> >::type storage_type;

    // Definition of the actual data fields that are used for input/output
    storage_type in(d1,d2,d3,-1, std::string("in"));
    storage_type out(d1,d2,d3,-7.3, std::string("out"));
    storage_type coeff(d1,d2,d3,8, std::string("coeff"));

    for (int i=0; i<d1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<d3; ++k) {
                in(i,j,k) = 0;
                out(i,j,k) = 0;
            }
        }
    }

#ifndef NDEBUG
    for (int i=0; i<d1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
    halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
    halos[2] = gridtools::halo_descriptor(1,1,1,d3-2,d3);

#ifdef CUDA_EXAMPLE
    in.clone_to_gpu();
    out.clone_to_gpu();
    in.h2d_update();
    out.h2d_update();

    gridtools::boundary_apply_gpu<bc_basic, minus_predicate>(halos, bc_basic(), minus_predicate()).apply(in);

    in.d2h_update();
#else
    gridtools::boundary_apply<bc_basic, minus_predicate>(halos, bc_basic(), minus_predicate()).apply(in);
#endif

#ifndef NDEBUG
    for (int i=0; i<d1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    bool result = true;

    for (int i=0; i<d1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<1; ++k) {
                if (in(i,j,k) != 0) {
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
                    result = false;
                }
            }
        }
    }

    for (int i=1; i<d1; ++i) {
        for (int j=1; j<d2; ++j) {
            for (int k=d3-1; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
                    result = false;
                }
            }
        }
    }

    for (int i=0; i<d1; ++i) {
        for (int j=0; j<1; ++j) {
            for (int k=0; k<d3; ++k) {
                if (in(i,j,k) != 0) {
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
                    result = false;
                }
            }
        }
    }

    for (int i=1; i<d1; ++i) {
        for (int j=d2-1; j<d2; ++j) {
            for (int k=1; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
                    result = false;
                }
            }
        }
    }

    for (int i=0; i<1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<d3; ++k) {
                if (in(i,j,k) != 0) {
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
                    result = false;
                }
            }
        }
    }

    for (int i=d1-1; i<d1; ++i) {
        for (int j=1; j<d2; ++j) {
            for (int k=1; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
                    result = false;
                }
            }
        }
    }

    for (int i=1; i<d1-1; ++i) {
        for (int j=1; j<d2-1; ++j) {
            for (int k=1; k<d3-1; ++k) {
                if (in(i,j,k) != 0) {
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
                    result = false;
                }
            }
        }
    }
    
    return result;

}