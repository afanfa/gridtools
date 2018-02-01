/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cassert>
#include <unordered_map>
#include <boost/program_options.hpp>

std::string add_define(std::string macro, int value) {
    std::string def;
    def += "#ifdef " + macro + "\n";
    def += "#if " + macro + " < " + std::to_string(value) + "\n";
    def += "#undef " + macro + "\n";
    def += "#define " + macro + " " + std::to_string(value) + "\n";
    def += "#endif\n#else\n";
    def += "#define " + macro + " " + std::to_string(value) + "\n";
    def += "#endif\n\n";
    return def;
}

struct prelude {
    std::string out() const {

        std::string code;
        code += "#include \"gtest/gtest.h\"\n";
        code += "#include <boost/mpl/equal.hpp>\n";
        code += "#include <stencil-composition/stencil-composition.hpp>\n";
        code += "\n";
        code += "#include \"backend_select.hpp\"\n";
        code += "\n";
        // code += "using namespace gridtools::enumtype;\n";
        // code += "using gridtools::accessor;\n";
        // code += "using gridtools::extent;\n";
        // code += "using gridtools::layout_map;\n";
        // code += "using gridtools::float_type;\n";
        // code += "using gridtools::arg;\n";
        // code += "using gridtools::uint_t;\n";
        // code += "using gridtools::int_t;\n";
        code += "\n";
        code += "using namespace gridtools;\n";
        code += "using namespace enumtype;\n";
        code += "\n";
        code += "typedef interval<level<0,-1>, level<1,-1> > x_interval;\n";
        code += "struct print_r {\n";
        code += "    template <typename T>\n";
        code += "    void operator()(T const& ) const {\n";
        code += "        std::cout << typename T::first() << \" \" << typename T::second() << std::endl;\n";
        code += "    }\n";
        code += "};\n\n";

        return code;
    }
};

struct range {
    int im, ip, jm, jp, km, kp;

    range() : im(-666), ip(666), jm(-666), jp(666), km(-666), kp(666) {}

    range(int im, int ip, int jm, int jp, int km, int kp) : im(im), ip(ip), jm(jm), jp(jp), km(km), kp(kp) {}

    range(range const &o) : im(o.im), ip(o.ip), jm(o.jm), jp(o.jp), km(o.km), kp(o.kp) {}

    range &operator=(range const &o) {
        im = o.im;
        ip = o.ip;
        jm = o.jm;
        jp = o.jp;
        km = o.km;
        kp = o.kp;
        return *this;
    }

    std::string out() const {
        std::string r = "extent<" + std::to_string(im) + ", " + std::to_string(ip) + ", " + std::to_string(jm) + ", " +
                        std::to_string(jp) + ", " + std::to_string(km) + ", " + std::to_string(kp) + "> ";
        return r;
    }
};

range operator+(range const &a, range const &b) {
    return range(a.im + b.im, a.ip + b.ip, a.jm + b.jm, a.jp + b.jp, a.km + b.km, a.kp + b.kp);
}

range operator||(range const &a, range const &b) {
    return range(std::min(a.im, b.im),
        std::max(a.ip, b.ip),
        std::min(a.jm, b.jm),
        std::max(a.jp, b.jp),
        std::min(a.km, b.km),
        std::max(a.kp, b.kp));
}

struct generate_functor {

    std::string m_name;
    int m_n_args;
    int m_index_of_output;
    std::vector< range > m_ranges;

    generate_functor() : m_name("not initialized"), m_n_args(-1), m_index_of_output(-1), m_ranges() {}

    template < typename RandG >
    generate_functor(std::string const &name, int n_args, int index_of_output, RandG gen)
        : m_name(name), m_n_args(n_args), m_index_of_output(index_of_output), m_ranges(m_n_args) {
        generate_ranges(gen);
    }

    generate_functor(generate_functor const &o)
        : m_name(o.m_name), m_n_args(o.m_n_args), m_index_of_output(o.m_index_of_output), m_ranges(o.m_ranges) {}

    template < typename RandGen >
    void generate_ranges(RandGen gen) {
        std::uniform_int_distribution<> out_gen(0, 3);

        for (int i = 0; i < m_n_args; ++i) {
            if (i == m_index_of_output) {
                m_ranges[i] = range(0, 0, 0, 0, 0, 0);
            } else {
                m_ranges[i] =
                    range(-out_gen(gen), out_gen(gen), -out_gen(gen), out_gen(gen), -out_gen(gen), out_gen(gen));
            }
        }
    }

    std::string name() const { return m_name; }
    int n_args() const { return m_n_args; }
    int index_of_output() const { return m_index_of_output; }

    range get_range(int i) const { return m_ranges[i]; }

    std::string out() const {
        std::string code = "";
        code += "struct " + m_name + "{\n";

        for (int i = 0; i < m_n_args; ++i) {
            if (i == m_index_of_output) {
                code += "    typedef accessor<" + std::to_string(i) + ", enumtype::inout> out;\n";
            } else {
                code += "    typedef accessor<" + std::to_string(i) + ", enumtype::in, " + m_ranges[i].out() + "> in" +
                        std::to_string(i) + ";\n";
            }
        }
        code += "\n    typedef boost::mpl::vector<"; // in,out> arg_list;

        for (int i = 0; i < m_n_args; ++i) {
            if (i == m_index_of_output) {
                code += "out";
            } else {
                code += "in" + std::to_string(i);
            }
            if (i < m_n_args - 1) {
                code += ",";
            } else {
                code += "> arg_list;\n";
            }
        }
        code += "\n    template <typename Evaluation>\n";
        code += "    GT_FUNCTION\n";
        code += "    static void Do(Evaluation & eval, x_interval) {}\n";
        code += "};\n";

        return code;
    }
};

std::ostream &operator<<(std::ostream &s, generate_functor const &f) { return s << f.out(); }
std::ostream &operator<<(std::ostream &s, prelude const &f) { return s << f.out(); }
std::ostream &operator<<(std::ostream &s, range const &f) { return s << f.out(); }

/** graph is made in tiers, one per functor, and the nodes are laied down
    as the arguments. One node per tier is the output one.
    So, in fact, the graph is the vector of functors already computed.
    But we need a place to put names.
*/
struct field_names {
    std::vector< std::vector< std::string > > m_names;

    template < typename FunctorVec >
    field_names(FunctorVec const &fv)
        : m_names(fv.size()) {
        for (int i = 0; i < m_names.size(); ++i) {
            m_names[i] = std::vector< std::string >(fv[i].n_args());
            for (int j = 0; j < m_names[i].size(); ++j) {
                m_names[i][j] = "na";
            }
        }
    }

    std::vector< std::string > &operator[](int i) { return m_names[i]; }

    int size() const { return m_names.size(); }

    std::string out() const {
        std::string o;
        for (int i = 0; i < m_names.size(); ++i) {
            o += "\nFunctor " + std::to_string(i) + "\n";
            for (int j = 0; j < m_names[i].size(); ++j) {
                o += m_names[i][j] + ", ";
            }
        }

        return o;
    }
};

int find_input_close_to(int idx, generate_functor const &functor, std::vector< std::string > const &names) {
    int initial = idx;

    while (idx == functor.index_of_output() || names[idx] != "na") {
        idx = (idx + 1) % functor.n_args();
        if (idx == initial) {
            return -1;
        }
    }
    return idx;
}

int main(int argc, char **argv) {

    bool make_comp, explicit_extents;
    unsigned seed;
    int ops;

    boost::program_options::options_description desc("Usage");
    desc.add_options()("make,m",
        boost::program_options::value< bool >(&make_comp)->default_value(false),
        "It 1/yes/on/true the code will actually run make_computation instead of simply compiute_extents. This is "
        "useful to benchmark the compiler. The check for correctness of the extents is turned off if this option is "
        "set to 1/yes/on/true\n")("explicit,e",
        boost::program_options::value< bool >(&explicit_extents)->default_value(false),
        "If -m or --make is specified, this option tells if the make_computation should use explicit extents")("seed,s",
        boost::program_options::value< unsigned >(&seed)->default_value(0),
        "Random seed for the random number generation. A vlaue equal to 0 will let the seed unspecified")("ops,o",
        boost::program_options::value< int >(&ops)->default_value(-1),
        "Number of operators (<=0 for random)")("help,h", "Produce help");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    if (seed != 0) {
        gen.seed(seed);
    }

    std::uniform_int_distribution<> functor_gen(1, 8);
    std::uniform_int_distribution<> arg_gen(2, 6);

    std::vector< generate_functor > functors((ops <= 0) ? functor_gen(gen) : ops);

    // generating arguments + output index
    int n = 0;
    for (auto &&i : functors) {
        int n_args = arg_gen(gen);
        std::uniform_int_distribution<> out_gen(0, n_args - 1);
        int output = out_gen(gen);
        std::string name = "functor" + std::to_string(n);
        i = generate_functor(name, n_args, output, gen);
        ++n;
    }

    std::string program = prelude().out();

    for (auto i : functors) {
        program += i.out() + "\n";
    }

    // int min_input_args = std::accumulate(functors.begin(), functors.end(), int(0),
    //                                      [](int a, generate_functor const& b) { return std::max(a,b.n_args()-1); });

    // int max_input_args = std::accumulate(functors.begin(), functors.end(), int(0),
    //                                      [](int a, generate_functor const& b) { return a+b.n_args()-1; });

    // std::cout << "Min Args " << min_input_args << std::endl;
    // std::cout << "Max Args " << max_input_args << std::endl;

    // Need a graph for dependencies.

    field_names names(functors);

    // output fields have unique names
    for (int i = 0; i < functors.size(); ++i) {
        std::string name = "o" + std::to_string(i);
        names[i][functors[i].index_of_output()] = name;
    }

    // fixing dependencies
    std::uniform_real_distribution< double > prob_range(0, 1);
    for (int i = 0; i < functors.size(); ++i) {
        double prob = 0.9;
        for (int j = i + 1; j < functors.size(); ++j) {
            if (prob_range(gen) < prob) {
                std::uniform_int_distribution<> idx_gen(0, functors[j].n_args() - 1);
                int index = find_input_close_to(idx_gen(gen), functors[j], names[j]);
                if (index != -1) {
                    names[j][index] = "o" + std::to_string(i);
                }
                prob /= 2.0;
            }
        }
    }

    // count non-assigned names
    int max_count = 0;
    int min_count = 0;
    for (int i = 0; i < names.size(); ++i) {
        int local_count = 0;
        for (int j = 0; j < names[i].size(); ++j) {
            if (names[i][j] == "na") {
                ++max_count;
                ++local_count;
            }
        }
        if (local_count > min_count)
            min_count = local_count;
    }

    assert(min_count <= max_count);
    std::uniform_int_distribution<> input_generators(min_count, max_count);
    std::vector< std::string > input_names(input_generators(gen));

    for (int i = 0; i < input_names.size(); ++i) {
        input_names[i] = "in" + std::to_string(i);
    }

    int iter = 0;
    for (int i = 0; i < names.size(); ++i) {
        for (int j = 0; j < names[i].size(); ++j) {
            if (names[i][j] == "na") {
                names[i][j] = input_names[iter];
                iter = (iter + 1) % input_names.size();
            }
        }
    }

    //   std::cout << names.out() << std::endl;

    // operators for debugging
    program += "\n";
    for (auto const &i : functors) {
        program +=
            "std::ostream& operator<<(std::ostream& s, " + i.name() + ") { return s << \"" + i.name() + "\"; }\n";
    }

    // boilerplate
    program += "gridtools::storage_traits< backend_t::s_backend_id > storage_tr;\n";
    program += "using storage_info_type = storage_tr::storage_info_t< 0, 3, gridtools::halo< 2, 2, 0 > >;\n";
    program += "using storage_type = storage_tr::data_store_t< gridtools::float_type, storage_info_type >;\n";
    program += "\n";
    program += "\n";

    std::string list_of_plcs = "typedef boost::mpl::vector<";

    // defining placeholders
    int global_index = 0;
    for (int i = 0; i < functors.size(); ++i) {
        program +=
            "typedef arg<" + std::to_string(global_index) + ", storage_type> o" + std::to_string(global_index) + ";\n";
        list_of_plcs += "o" + std::to_string(global_index) + ", ";
        ++global_index;
    }

    for (int i = 0; i < input_names.size(); ++i) {
        program += "typedef arg<" + std::to_string(global_index) + ", storage_type> " + input_names[i] + ";\n";
        list_of_plcs += input_names[i];
        if (i != input_names.size() - 1) {
            list_of_plcs += ", ";
        }
        ++global_index;
    }

    list_of_plcs += "> placeholders;\n";

    /********************************************************************************
     This is the part that computes the extents given the computation to the check
    ********************************************************************************/
    std::unordered_map< std::string, range > map;

    for (int i = 0; i < functors.size(); ++i) {
        map.insert({"o" + std::to_string(i), range(0, 0, 0, 0, 0, 0)});
    }
    for (int i = 0; i < input_names.size(); ++i) {
        map.insert({input_names[i], range(0, 0, 0, 0, 0, 0)});
    }

    for (int i = functors.size() - 1; i >= 0; --i) {
        std::string out_name = names[i][functors[i].index_of_output()];
        // std::cout << "/* " << functors[i].index_of_output() << " ********* " << out_name << " */" << std::endl;
        range out_range = map[out_name];
        for (int j = 0; j < functors[i].n_args(); ++j) {
            if (j != functors[i].index_of_output()) {
                // std::cout << names[i][j] << std::endl;
                // std::cout << out_range.out() << std::endl;
                range updated_range = functors[i].get_range(j) + out_range;
                // std::cout << updated_range.out() << std::endl;
                updated_range = updated_range || map[names[i][j]];
                // std::cout << updated_range.out() << std::endl;
                map[names[i][j]] = updated_range;
            }
        }
    }
    /********************************************************************************
     End of the part that computes the extents given the computation to the check
    ********************************************************************************/

    // additional boilerplate
    program += "int main() {\n";

    for (int i = 0; i < functors.size(); ++i) {
        if (explicit_extents) {
            program += "    typedef decltype(make_stage_with_extent<" + functors[i].name() + ", " +
                       map["o" + std::to_string(i)].out() + ">(";
        } else {
            program += "    typedef decltype(make_stage<" + functors[i].name() + ">(";
        }
        for (int j = 0; j < names[i].size(); ++j) {
            program += names[i][j] + "()";
            if (j != names[i].size() - 1) {
                program += ", ";
            }
        }
        program += ")) " + functors[i].name() + "__;\n";
    }

    program += "    auto mss = make_multistage\n";
    program += "        (\n";
    program += "            execute<forward>(),\n";
    for (int i = 0; i < functors.size(); ++i) {
        program += "            " + functors[i].name() + "__()";
        if (i != functors.size() - 1) {
            program += ",\n";
        }
    }
    program += "        );\n";
    program += "    using mss_t = decltype(mss);\n";

    program += "    " + list_of_plcs;

    if (make_comp) {
        program += "    storage_info_type mock_info(3,3,3);\n";
        program += "    storage_type mock_stor(mock_info, \"mock_storage\");\n";

        std::string agg = "    aggregator_type< placeholders > agg(";

        for (int i = 0; i < functors.size(); ++i) {
            agg += "(o" + std::to_string(i) + "() = mock_stor), ";
        }

        for (int i = 0; i < input_names.size(); ++i) {
            agg += "(" + input_names[i] + "() = mock_stor)";
            if (i != input_names.size() - 1) {
                agg += ", ";
            }
        }
        program += agg + ");\n";

        program += "    uint_t di[5] = {1,1,1,3,5};\n";
        program += "    uint_t dj[5] = {1,1,1,3,5};\n";
        program += "    typedef gridtools::interval< level< 0, -1 >, level< 1, 1 > > axis;\n";
        program += "    grid< axis > grid(di, dj);\n";
        program += "    grid.value_list[0] = 0;\n";
        program += "    grid.value_list[1] = 3;\n\n";
        program += "    auto stencil = make_computation<backend_t>(\n";
        program += "        agg,\n";
        program += "        grid,\n";
        program += "        mss);\n";
    } else {
        program += "\n    typedef "
                   "compute_extents_of<init_map_of_extents<placeholders>::type,1>::for_mss<mss_t>::type "
                   "final_map;\n";

        program += "    std::cout << \"FINAL\" << std::endl;\n";
        program += "    boost::mpl::for_each<final_map>(print_r());\n\n";

        for (int i = 0; i < functors.size(); ++i) {
            program += "GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, o" + std::to_string(i) +
                       ">::type, " + map["o" + std::to_string(i)].out() + ">::type::value),\n";
            program += "                          \"o" + std::to_string(i) + " " + map["o" + std::to_string(i)].out() +
                       "\");\n";
        }
        for (int i = 0; i < input_names.size(); ++i) {
            program += "GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, " + input_names[i] +
                       ">::type, " + map[input_names[i]].out() + ">::type::value),\n";
            program += "                          \"" + input_names[i] + " " + map[input_names[i]].out() + "\");\n";
        }
    }

    int total_placeholders = functors.size() + input_names.size();
    if ((total_placeholders / 10) * 10 != total_placeholders) {
        total_placeholders = (total_placeholders / 10 + 1) * 10;
    }

    program += "/* total placeholders (rounded to 10) _SIZE = " + std::to_string(total_placeholders) + "*/\n";

    if (total_placeholders > 20) { // Adding macros in reverse!
        std::string defines;
        defines += add_define("BOOST_MPL_LIMIT_VECTOR_SIZE", total_placeholders);
        defines += add_define("BOOST_MPL_LIMIT_MAP_SIZE", total_placeholders);
        defines += add_define("FUSION_MAX_VECTOR_SIZE", total_placeholders);
        defines += add_define("FUSION_MAX_MAP_SIZE", total_placeholders);
        defines += add_define("BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS", total_placeholders);

        program = defines + program;
    }

    program += "    return 0;\n";
    program += "}\n";

    std::cout << program << std::endl;
    return 0;
}