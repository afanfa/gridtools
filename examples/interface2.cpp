#include <gridtools.h>
#ifdef CUDA_EXAMPLE
#include <backend_cuda.h>
#else
#include <backend_block.h>
#include <backend_naive.h>
#endif

#include <stdlib.h>

/*
  This file shows an implementation of the "horizontal diffusion" stencil, similar to the one used in COSMO
 */

using gridtools::level;
using gridtools::arg_type;
using gridtools::range;
using gridtools::arg;

// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_lap;
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_flx;
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_out;

typedef gridtools::interval<level<0,-2>, level<1,3> > axis;

// These are the stencil operators that compose the multistage stencil in this test
struct lap_function {
    static const int n_args = 2;
    typedef arg_type<0> out;
    typedef const arg_type<1, range<-1, 1, -1, 1> > in;
    typedef typename boost::mpl::vector<out, in> arg_list;

    template <typename t_domain>
    GT_FUNCTION
    static void Do(t_domain const & dom, x_lap) {
        dom(out()) = 3*dom(in()) -
            (dom(in( 1, 0, 0)) + dom(in( 0, 1, 0)) +
             dom(in(-1, 0, 0)) + dom(in( 0,-1, 0)));
    }
};

struct flx_function {
    static const int n_args = 3;
    typedef arg_type<0> out;
    typedef const arg_type<1, range<0, 1, 0, 0> > in;
    typedef const arg_type<2, range<0, 1, 0, 0> > lap;

    typedef typename boost::mpl::vector<out, in, lap> arg_list;

    template <typename t_domain>
    GT_FUNCTION
    static void Do(t_domain const & dom, x_flx) {
        dom(out()) = dom(lap(1,0,0))-dom(lap(0,0,0));
        if (dom(out())*(dom(in(1,0,0))-dom(in(0,0,0)))) {
            dom(out()) = 0.;
        }
    }
};

struct fly_function {
    static const int n_args = 3;
    typedef arg_type<0> out;
    typedef const arg_type<1, range<0, 0, 0, 1> > in;
    typedef const arg_type<2, range<0, 0, 0, 1> > lap;
    typedef typename boost::mpl::vector<out, in, lap> arg_list;

    template <typename t_domain>
    GT_FUNCTION
    static void Do(t_domain const & dom, x_flx) {
        dom(out()) = dom(lap(0,1,0))-dom(lap(0,0,0));
        if (dom(out())*(dom(in(0,1,0))-dom(in(0,0,0)))) {
            dom(out()) = 0.;
        }
    }
};

struct out_function {
    static const int n_args = 5;
    typedef arg_type<0> out;
    typedef const arg_type<1> in;
    typedef const arg_type<2, range<-1, 0, 0, 0> > flx;
    typedef const arg_type<3, range<0, 0, -1, 0> > fly;
    typedef const arg_type<4> coeff;
    typedef typename boost::mpl::vector<out,in,flx,fly,coeff> arg_list;

    template <typename t_domain>
    GT_FUNCTION
    static void Do(t_domain const & dom, x_out) {
        dom(out()) = dom(in()) - dom(coeff()) *
            (dom(flx()) - dom(flx(-1,0,0)) +
             dom(fly()) - dom(fly(0,-1,0))
             );
    }
};

/*
 * The following operators and structs are for debugging only
 */
std::ostream& operator<<(std::ostream& s, lap_function const) {
    return s << "lap_function";
}
std::ostream& operator<<(std::ostream& s, flx_function const) {
    return s << "flx_function";
}
std::ostream& operator<<(std::ostream& s, fly_function const) {
    return s << "fly_function";
}
std::ostream& operator<<(std::ostream& s, out_function const) {
    return s << "out_function";
}

int main(int argc, char** argv) {
    int d1 = atoi(argv[1]);
    int d2 = atoi(argv[2]);
    int d3 = atoi(argv[3]);

#ifdef CUDA_EXAMPLE
#define STORAGE cuda_storage
#else
#define STORAGE storage
#endif

    typedef gridtools::STORAGE<double, gridtools::layout_map<0,1,2> > storage_type;

     // Definition of the actual data fields that are used for input/output
    storage_type in(d1,d2,d3,-1, std::string("in"));
    storage_type out(d1,d2,d3,-7.3, std::string("out"));
    storage_type coeff(d1,d2,d3,-3, std::string("coeff"));

    out.print();

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain
    typedef arg<3, gridtools::temporary<storage_type> > p_lap;
    typedef arg<2, gridtools::temporary<storage_type> > p_flx;
    typedef arg<4, gridtools::temporary<storage_type> > p_fly;
    typedef arg<5, storage_type > p_coeff;
    typedef arg<1, storage_type > p_in;
    typedef arg<0, storage_type > p_out;

    // An array of placeholders to be passed to the domain
    // I'm using mpl::vector, but the final API should look slightly simpler
    typedef boost::mpl::vector<p_lap, p_flx, p_fly, p_coeff, p_in, p_out> arg_type_list;

    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
    gridtools::domain_type<arg_type_list> domain
        (boost::fusion::make_vector(&out, &in, &coeff /*,&fly, &flx*/));

    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    gridtools::coordinates<axis> coords(2,d1-2,2,d2-2);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3;

#ifdef CUDA_EXAMPLE
#define BACKEND backend_cuda
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend_block
#else
#define BACKEND backend_naive
#endif
#endif

    /*
      Here we do lot of stuff
      1) We pass to the intermediate representation ::run function the description
      of the stencil, which is a multi-stage stencil (mss)
      The mss includes (in order of execution) a laplacian, two fluxes which are independent
      and a final step that is the out_function
      2) The logical physical domain with the fields to use
      3) The actual domain dimensions
     */
    // gridtools::intermediate::run<gridtools::BACKEND>
    //     (
    //      gridtools::make_mss
    //      (
    //       gridtools::execute_upward,
    //       gridtools::make_esf<lap_function>(p_lap(), p_in()),
    //       gridtools::make_independent
    //       (
    //        gridtools::make_esf<flx_function>(p_flx(), p_in(), p_lap()),
    //        gridtools::make_esf<fly_function>(p_fly(), p_in(), p_lap())
    //        ),
    //       gridtools::make_esf<out_function>(p_out(), p_in(), p_flx(), p_fly(), p_coeff())
    //       ),
    //      domain, coords);

    gridtools::computation *horizontal_diffusion =
        gridtools::make_computation<gridtools::BACKEND>
        (
         gridtools::make_mss
         (
          gridtools::execute_upward,
          gridtools::make_esf<lap_function>(p_lap(), p_in()),
          gridtools::make_independent
          (
           gridtools::make_esf<flx_function>(p_flx(), p_in(), p_lap()),
           gridtools::make_esf<fly_function>(p_fly(), p_in(), p_lap())
           ),
          gridtools::make_esf<out_function>(p_out(), p_in(), p_flx(), p_fly(), p_coeff())
          ),
         domain, coords);

    horizontal_diffusion->setup();
    horizontal_diffusion->prepare();
    horizontal_diffusion->run();
    horizontal_diffusion->finalize();



    //    in.print();
    out.print();
    //    lap.print();

    return 0;
}
