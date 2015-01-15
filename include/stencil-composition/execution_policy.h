#pragma once
#include "basic_token_execution.h"
#ifdef __CUDACC__
#include "cuda_profiler_api.h"
#endif

/**
@file Implementation of the k loop execution policy
The policies which are currently considered are
 - forward: the k loop is executed upward, increasing the value of the iterator on k. This is the option to be used when the stencil operations at level k depend on the fields at level k-1 (forward substitution).
 - backward: the k loop is executed downward, decreasing the value of the iterator on k. This is the option to be used when the stencil operations at level k depend on the fields at level k+1 (backward substitution).
 - parallel: the operations on each k level are executed in parallel. This is feasable only if there are no dependencies between levels.
*/

namespace gridtools{
    namespace _impl{

/**
   @brief   Execution kernel containing the loop over k levels
*/
        template< typename ExecutionEngine, typename ExtraArguments >
        struct run_f_on_interval{
            typedef uint_t local_domain_t;
        };

/**
   @brief partial specialization for the forward or backward cases
*/
        template<
            enumtype::execution IterationType,
            typename ExtraArguments>
        struct run_f_on_interval< typename enumtype::execute<IterationType>, ExtraArguments > : public run_f_on_interval_base< run_f_on_interval<typename enumtype::execute<IterationType>, ExtraArguments > >
        {
	    typedef run_f_on_interval_base< run_f_on_interval<typename enumtype::execute<IterationType>, ExtraArguments > > super;
            typedef typename enumtype::execute<IterationType>::type execution_engine;
            typedef ExtraArguments traits;

            GT_FUNCTION
            explicit run_f_on_interval(typename traits::local_domain_t & domain, typename traits::coords_t const& coords):super(domain, coords){}


            template<typename IterationPolicy, typename IntervalType>
            GT_FUNCTION
            void loop(uint_t from, uint_t to) const {

	      for ( uint_t k=from ; k<=to; ++k, IterationPolicy::increment(this->m_domain)) {
	      	traits::functor_t::Do(this->m_domain, IntervalType());
		/* printf("k=%d\n", k); */
	      }

            }
        };

/**
   @brief partial specialization for the parallel case (to be implemented)
   stub
*/
        // template<
        //     typename ExtraArguments>
        // struct run_f_on_interval<typename enumtype::execute<enumtype::parallel>, ExtraArguments > : public run_f_on_interval_base< run_f_on_interval<typename enumtype::execute<enumtype::parallel>, ExtraArguments > >
        // {
	//     exit(-37);
        // };
    } // namespace _impl
} // namespace gridtools