#pragma once

#include "../common/defs.h"
#include "../common/gt_assert.h"

#include <stdio.h>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/fusion/view/filter_view.hpp>
#include <boost/fusion/include/for_each.hpp>
#include "gt_for_each/for_each.hpp"

#include "domain_type_impl.h"

namespace gridtools {

    /**
     * @tparam Placeholders list of placeholders of type arg<I,T>
     */
    template <typename Placeholders>
    struct domain_type : public clonable_to_gpu<domain_type<Placeholders> > {
        typedef Placeholders original_placeholders;
    private:
        BOOST_STATIC_CONSTANT(int, len = boost::mpl::size<original_placeholders>::type::value);

        /**
         * \brief Get a sequence of the same type of original_placeholders, but containing the storage types for each placeholder
         * \todo I would call it instead of l_get_type l_get_storage_type
         */
        typedef typename boost::mpl::transform<original_placeholders,
                                               _impl::l_get_type
                                               >::type raw_storage_list;

        /**
         * \brief Get a sequence of the same type of original_placeholders, but containing the iterator types corresponding to the placeholder's storage type
         */
        typedef typename boost::mpl::transform<original_placeholders,
                                               _impl::l_get_it_type
                                               >::type raw_iterators_list;

    public:
        /**
         * \brief Get a sequence of the same type as original_placeholders, containing the indexes relative to the placehoolders
         * note that the static const indexes are transformed into types using mpl::integral_c
         */
        typedef typename boost::mpl::transform<original_placeholders,
                                               _impl::l_get_index
                                               >::type raw_index_list;

        /**
         * \brief Definition of a random access sequence of integers between 0 and the size of the placeholder sequence
         e.g. [0,1,2,3,4]
         */
        typedef boost::mpl::range_c<int,0,len> range_t;
    private:

        /**\brief reordering vector
         * defines an mpl::vector of len indexes reordered accodring to range_t (placeholder _2 is vector<>, placeholder _1 is range_t)
         e.g.[1,3,2,4,0]
         */
        typedef typename boost::mpl::fold<range_t,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1,
                                              boost::mpl::find<raw_index_list, boost::mpl::_2>
                                              >
                                          >::type iter_list;

    public:

        /**\brief reordered index_list
         * Defines a mpl::vector of index::pos for the indexes in iter_list
         */
        typedef typename boost::mpl::transform<iter_list,
                                               _impl::l_get_it_pos
                                               >::type index_list;

        /**
         * \brief reordering of raw_storage_list
         creates an mpl::vector of all the storages in raw_storage_list corresponding to the indices in index_list
         */
        typedef typename boost::mpl::fold<index_list,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1,
                                              boost::mpl::at<raw_storage_list, boost::mpl::_2>
                                              >
                                          >::type arg_list_mpl;

        /**
         * \brief defines a reordered mpl::vector of placeholders
         */
        typedef typename boost::mpl::fold<index_list,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1,
                                              boost::mpl::at<original_placeholders, boost::mpl::_2>
                                              >
                                          >::type placeholders;

    private:
        typedef typename boost::mpl::fold<index_list,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1,
                                              boost::mpl::at<raw_iterators_list, boost::mpl::_2> >
                                          >::type iterator_list_mpl;

    public:
        /**
         * Type of fusion::vector of pointers to storages as indicated in Placeholders
         */
        typedef typename boost::fusion::result_of::as_vector<arg_list_mpl>::type arg_list;
        /**
         * Type of fusion::vector of pointers to iterators as indicated in Placeholders
         */
        typedef typename boost::fusion::result_of::as_vector<iterator_list_mpl>::type iterator_list;

        /**
         * fusion::vector of pointers to storages
         */
        arg_list storage_pointers;

        /**
         * fusion::vector of pointers to storages before the updates needed before the start of the computation
         */
        arg_list original_pointers;

    public:

        /**
         * @tparam RealStorage fusion::vector of pointers to storages sorted with increasing indices of the pplaceholders
         * @param real_storage The actual fusion::vector with the values
         */
        template <typename RealStorage>
        explicit domain_type(RealStorage const & real_storage)
         : storage_pointers()
        {

#ifndef NDEBUG
            std::cout << "These are the original placeholders and their storages" << std::endl;
            gridtools::for_each<original_placeholders>(_debug::stdcoutstuff());
#endif

            typedef boost::fusion::filter_view<arg_list,
                is_storage<boost::mpl::_1> > view_type;

            view_type fview(storage_pointers);

            BOOST_MPL_ASSERT_MSG( (boost::fusion::result_of::size<view_type>::type::value == boost::mpl::size<RealStorage>::type::value), _NUMBER_OF_ARGS_SEEMS_WRONG_, (boost::fusion::result_of::size<view_type>) );

#ifndef NDEBUG
            // std::cout << "These are the actual placeholders and their storages" << std::endl;
            // gridtools::for_each<placeholders>(_debug::stdcoutstuff());
            std::cout << "These are the real storages" << std::endl;
            boost::fusion::for_each(real_storage, _debug::print_deref());
            std::cout << "\nThese are the arg_list elems" << std::endl;
            boost::fusion::for_each(arg_list(), _debug::print_deref());
            std::cout << "\nThese are the storage_pointers elems" << std::endl;
            boost::fusion::for_each(arg_list(), _debug::print_deref());
            std::cout << "\nThese are the view " << boost::fusion::size(fview) << std::endl;
            boost::fusion::for_each(fview, _debug::print_deref());
#endif
            boost::fusion::copy(real_storage, fview);

            view_type original_fview(original_pointers);
            boost::fusion::copy(real_storage, original_fview);
        }

#ifdef __CUDACC__
        /** Copy constructor to be used when cloning to GPU
         *
         * @param The object to copy. Typically this will be *this
         */
        __device__
        explicit domain_type(domain_type const& other)
            : storage_pointers(other.storage_pointers)
            , original_pointers(other.original_pointers)
        { }
#endif

        GT_FUNCTION
        void info() {
            // printf("domain_type: Storage pointers\n");
            // boost::fusion::for_each(storage_pointers, _debug::print_domain_info());
            // printf("domain_type: Original pointers\n");
            // boost::fusion::for_each(original_pointers, _debug::print_domain_info());
            // printf("domain_type: End info\n");
        }

        template <typename Index>
        void storage_info() const {
            // std::cout << Index::value << " -|-> "
            //           << (boost::fusion::at_c<Index::value>(storage_pointers))->name()
            //           << " "
            //           << (boost::fusion::at_c<Index::value>(storage_pointers))->m_dims[0]
            //           << "x"
            //           << (boost::fusion::at_c<Index::value>(storage_pointers))->m_dims[1]
            //           << "x"
            //           << (boost::fusion::at_c<Index::value>(storage_pointers))->m_dims[2]
            //           << ", "
            //           << (boost::fusion::at_c<Index::value>(storage_pointers))->strides[0]
            //           << "x"
            //           << (boost::fusion::at_c<Index::value>(storage_pointers))->strides[1]
            //           << "x"
            //           << (boost::fusion::at_c<Index::value>(storage_pointers))->strides[2]
            //           << ", "
            //           << std::endl;
        }

        // ~domain_type() {
        //     typedef boost::fusion::filter_view<arg_list,
        //         is_temporary_storage<boost::mpl::_> > tmp_view_type;
        //     tmp_view_type fview(storage_pointers);
        // }

        /** @brief copy the pointers from the hdevice to the host */
        void finalize_computation() {
            boost::fusion::for_each(original_pointers, _impl::call_d2h());
            boost::fusion::copy(original_pointers, storage_pointers);
        }

    };

} // namespace gridtools
