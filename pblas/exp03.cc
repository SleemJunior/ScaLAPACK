// Create a 10 by 10 matrix X[i,j] = 10*i + j and figure out how the thing is distributed.

#include <mkl.h>
#include <mkl_scalapack.h>
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mpi.h>

#include <iostream>
#include <vector>
#include <functional>
#include <tuple>

#include <chrono>

#define DCB_SETTING(x)  std::cout << x << std::endl
#define DCB_INIT(x) std::cout << x << std::endl;

using std::vector;
using std::function;
using std::tuple;

using namespace std::chrono;

typedef MKL_INT desc_t[ 9 ];
using mint_t = MKL_INT;
using dims_t = tuple<mint_t, mint_t>;

const float float_zero = 0.0;
const float float_one  = 1.0;

const mint_t mint_negative_one = -1;
const mint_t mint_one          = 1;
const mint_t mint_zero         = 0;
const char char_no_trans = 'N';
const char char_trans    = 'T';
const char char_row = 'R';
const char char_col = 'C';
const char char_all = 'A';

struct setting_t
{
  setting_t(dims_t const& grid_dims_): grid_dims(grid_dims_)
  {
    auto& [num_row, num_col] = grid_dims;

    auto& [current_row, current_col] = grid_location;
    current_row = -7;
    current_col = -7;

    blacs_pinfo( &rank, &num_ranks );
    if(num_ranks != num_row*num_col) {
      throw std::runtime_error("grid dims must equal the number of processes");
    }

    blacs_get(&mint_negative_one, &mint_zero, &handle);

    blacs_gridinit(&handle, &char_row, &num_row, &num_col);

    blacs_gridinfo(&handle, &num_row, &num_col, &current_row, &current_col);

    if(grid_dims != grid_dims_) {
      throw std::runtime_error("blacs_gridinfo not well behaved");
    }
    if(num_row <= 0 || num_col <= 0 || current_row < 0 || current_col < 0) {
      throw std::runtime_error("setting_t: couldn't get grid info");
    }
  }

  ~setting_t(){
    blacs_gridexit(&handle);
    blacs_exit(&mint_zero);
  }

  dims_t grid_dims;
  dims_t grid_location;

  mint_t rank;
  mint_t num_ranks;

  mint_t handle;
};

struct dist_mat_t
{
  dist_mat_t(
    setting_t const& setting_,
    dims_t const& dims_,
    dims_t const& blocks_)
      : setting(setting_), dims(dims_), blocks(blocks_)
  {
    auto const& [d0,d1] = dims;
    auto const& [b0,b1] = blocks;
    auto&       [l0,l1] = local_dims;

    auto const& [g0,   g1] = setting.grid_location;
    auto const& [ng0, ng1] = setting.grid_dims;

    l0 = numroc(&d0, &b0, &g0, &mint_zero, &ng0);
    l1 = numroc(&d1, &b1, &g1, &mint_zero, &ng1);

    if(l0 <= 0 || l1 <= 0) {
      throw std::runtime_error("l0 and l1 ar the wrong size");
    }

    data = vector<float>(l0*l1);

    mint_t success = -1;
    descinit(desc_local,
      &d0, &d1,
      &d0, &d1,
      &mint_zero, &mint_zero,
      &setting.handle,
      &d0,
      &success);
    if(success != 0) {
      throw std::runtime_error("descinit for desc_local failed");
    }

    success = -1;
    descinit(desc,
      &d0, &d1,
      &b0, &b1,
      &mint_zero, &mint_zero,
      &setting.handle,
      &l0,
      &success);
    if(success != 0) {
      throw std::runtime_error("descinit for desc (global) failed");
    }
  }

  // Don't copy the data
  dist_mat_t(dist_mat_t const& other) = delete;

  // Move the data
  dist_mat_t(dist_mat_t && other):
    setting(other.setting),
    local_dims(other.local_dims),
    dims(other.dims), blocks(other.blocks),
    data(std::move(other.data))
  {
    std::copy(other.desc,       other.desc       + 9, desc);
    std::copy(other.desc_local, other.desc_local + 9, desc_local);
  }


  void init() {
    if(setting.rank == 0) {
      throw std::runtime_error("don't call init() with processor zero");
    }
    auto& [d0,d1] = dims;
    psgeadd(
      &char_no_trans,
      &d0, &d1,
      &float_one,  nullptr,     &mint_one, &mint_one, desc_local,
      &float_zero, data.data(), &mint_one, &mint_one, desc);
  }

  void init(float* global_data) {
    if(setting.rank != 0) {
      throw std::runtime_error("don't call init(float*) with processor not-zero");
    }
    auto& [d0,d1] = dims;
    psgeadd(
      &char_no_trans,
      &d0, &d1,
      &float_one,  global_data, &mint_one, &mint_one, desc_local,
      &float_zero, data.data(), &mint_one, &mint_one, desc);
  }

  void init(function<vector<float>()> f) {
    if(setting.rank != 0) {
      return init();
    }

    auto vec = f();
    auto [d0,d1] = dims;
    if(vec.size() != d0*d1) {
      throw std::runtime_error("invalid init data size");
    }
    return init(vec.data());
  }

  void init(function<float(int,int)> f) {
    if(setting.rank != 0) {
      return init();
    }

    vector<float> d;
    auto [d0,d1] = dims;
    d.reserve(d0*d1);
    for(mint_t i = 0; i != d0; ++i) {
    for(mint_t j = 0; j != d1; ++j) {
      d.push_back(f(i,j));
    }}

    init(d.data());
  }

  void print() {
    blacs_barrier(&setting.handle, &char_all);
    auto const& [l0,l1] = local_dims;
    auto const& [g0,g1] = setting.grid_location;
    for(int rank = 0; rank != setting.num_ranks; ++rank) {
      if(setting.rank == rank) {
         std::cout << "-------------------------------------------" << std::endl;
         std::cout << "proc:        " << setting.rank  << std::endl;
         std::cout << "grid:       [" << g0 << ", " << g1 << "]" << std::endl;
         std::cout << "local dims: [" << l0 << ", " << l1 << "]" << std::endl;
         for(mint_t i = 0; i != l0; ++i) {
         for(mint_t j = 0; j != l1; ++j) {
           std::cout << data[i*l1 + j] << " ";
         }}
         std::cout << std::endl;
      }
      blacs_barrier(&setting.handle, &char_all);
    }
  }

  vector<float> data;

  setting_t const& setting;

  dims_t local_dims;
  dims_t dims;
  dims_t blocks;

  desc_t desc;
  desc_t desc_local;
};

dist_mat_t
matmul(dist_mat_t const& lhs, dist_mat_t const& rhs)
{
  auto const& [ni, nj] = lhs.dims;
  auto const& [nj_,nk] = rhs.dims;

  auto const& [bi, bj] = lhs.blocks;
  auto const& [bj_,bk] = rhs.blocks;

  if(nj != nj_ || bj != bj_) {
    throw std::runtime_error("invalid matmul");
  }

  dist_mat_t ret(lhs.setting, dims_t(ni,nk), dims_t(bi, bk));

  psgemm(
    &char_trans, &char_no_trans,
    &ni, &nk, &nj,
    &float_one,
    lhs.data.data(), &mint_one, &mint_one, lhs.desc,
    rhs.data.data(), &mint_one, &mint_one, rhs.desc,
    &float_zero,
    ret.data.data(), &mint_one, &mint_one, ret.desc);

  return std::move(ret);
}

using time_measurement_t = decltype(std::chrono::high_resolution_clock::now());
using time_difference_t =

struct barrier_timer_t {
  barrier_timer_t(setting_t const& setting, double& elapsed_time):
    setting(setting), elapsed_time(elapsed_time)
  {
    blacs_barrier(&setting.handle, &char_all);
    start = std::chrono::high_resolution_clock::now();
  }

  ~barrier_timer_t() {
    blacs_barrier(&setting.handle, &char_all);
    auto end = std::chrono::high_resolution_clock::now();
    elapsed_time = (double) duration_cast<microseconds>(end - start).count()
                      / (double) duration_cast<microseconds>(1s).count();
  }

  setting_t const& setting;
  time_measurement_t start;
  double& elapsed_time;
};

int main(int argc, char** argv) {
  mint_t nd       = atoi(argv[1]);

  mint_t n_rank_row = atoi(argv[2]);
  mint_t n_rank_col = atoi(argv[3]);

  dims_t grid_dims(n_rank_row, n_rank_col);
  setting_t setting(grid_dims);

  dims_t dims(nd,nd);
  dims_t blocks(nd / n_rank_row, nd / n_rank_col);

  auto f_random = [nd]() {
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MCG31, time(nullptr));

    vector<float> ret;
    ret.resize(nd*nd);

    vsRngUniform(
      VSL_RNG_METHOD_UNIFORM_STD,
      stream,
      nd*nd,
      ret.data(),
      -1.0 ,
       1.0);

    vslDeleteStream(&stream);

    return ret;
  };

  dist_mat_t A(setting, dims, blocks);
  dist_mat_t B(setting, dims, blocks);

  A.init(f_random);
  B.init(f_random);

  std::cout << "Starting." << std::endl;

  double elapsed_time = -1.0;
  {
    barrier_timer_t timer(setting, elapsed_time);
    dist_mat_t C = std::move(matmul(A, B));
  }

  if(setting.rank == 0) {
    std::cout <<
      "Dimension " << nd << std::endl <<
      "Processor grid [" << n_rank_row << ", " << n_rank_col << "] " << std::endl <<
      "Total time: " << elapsed_time << " seconds" <<
      std::endl;
  }
}
