// Create a 10 by 10 matrix X[i,j] = 10*i + j and figure out how the thing is distributed.

// Took a look at this:
//  https://github.com/cjf00000/tests/blob/master/mkl/compile.sh
//  https://github.com/cjf00000/tests/blob/master/mkl/pblas3_d_example.c

#include <mkl_scalapack.h>
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mpi.h>

#include <iostream>
#include <vector>
#include <functional>
#include <tuple>

#define _max(x, y) ((x)>(y) ? (x):(y))

#define DCB_SETTING(x)  std::cout << x << std::endl
#define DCB_INIT(x) std::cout << x << std::endl;

using std::vector;
using std::function;
using std::tuple;

typedef MKL_INT desc_t[ 9 ];
using mint_t = MKL_INT;
using dims_t = tuple<mint_t, mint_t>;

const float float_zero = 0.0;
const float float_one  = 0.0;

const mint_t mint_negative_one = -1;
const mint_t mint_one          = 1;
const mint_t mint_zero         = 0;
const mint_t mint_ten          = 10;
const char char_no_trans = 'N';
const char char_row = 'R';
const char char_col = 'C';

struct setting_t 
{
  setting_t(dims_t const& grid_dims_): grid_dims(grid_dims_)
  {
    auto& [num_row, num_col] = grid_dims;

    auto& [current_row, current_col] = grid_location;
    current_row = -7;
    current_col = -7;

    blacs_pinfo( &which, &num_processes );
    if(num_processes != num_row*num_col) {
      throw std::runtime_error("grid dims must equal the number of processes");
    }

    *handle = MPI_COMM_WORLD;

    blacs_get(&mint_negative_one, &mint_zero, handle);

    {
      auto nrow = num_row;
      auto ncol = num_col ;
      blacs_gridinit(handle, &char_row, &nrow, &ncol);
    }

    { // This is necessary and it is stupid.
      auto nrow = num_row;
      auto ncol = num_col;
      auto crow = current_row;
      auto ccol = current_col;
      blacs_gridinfo(handle, &nrow, &ncol, &crow, &ccol);
      current_row = crow;
      current_col = ccol;
    }
    if(grid_dims != grid_dims_) {
      throw std::runtime_error("blacs_gridinfo not well behaved");
    }
    if(num_row <= 0 || num_col <= 0 || current_row < 0 || current_col < 0) {
      throw std::runtime_error("setting_t: couldn't get grid info");
    }
    //DCB_SETTING("!! [" << current_row << ", " << current_col << "]");

  }

  ~setting_t(){ 
    blacs_gridexit(handle);
    blacs_exit(&mint_zero);
  }

  // The problem with this api is that everything is taken by pointers.
  // A previous bug: handle came before grid_dims and blacs_gridinit decided to 
  // write to the memory after handle which ended up overwriting grid_dims. Go figure.
  // So just give the handle a bunch of extra space.

  // Mysterious things still happening..

  dims_t grid_dims;
  dims_t grid_location;

  mint_t which;
  mint_t num_processes;

  mint_t handle[20];
};

struct dist_matmul_t 
{
  dist_matmul_t(
    setting_t const& setting_, 
    dims_t const& dims_,
    dims_t const& blocks_)
      : setting(setting_), dims(dims_), blocks(blocks_), data(nullptr)
  {
    auto const& [d0,d1] = dims;
    auto const& [b0,b1] = blocks;
    auto&       [l0,l1] = local_dims;

    auto const& [g0,   g1] = setting.grid_location;
    auto const& [ng0, ng1] = setting.grid_dims;

//    DCB_INIT("[" << g0 << ", " << g1 << "]");
//    DCB_INIT("[" << d0 << "," << d1 << "]");
//    DCB_INIT(d0 << ", " << b0 << ", " << g0 << ", zero " << mint_zero << ", " << ng0);
//    DCB_INIT(d0 << ", " << b0 << ", " << g0 << ", " << ng0);
//    DCB_INIT(d1 << ", " << b1 << ", " << g1 << ", " << ng1);
    l0 = numroc(&d0, &b0, &g0, &mint_zero, &ng0);
//    l1 = numroc(&d1, &b1, &g1, &mint_zero, &ng1);
//    DCB_INIT(d0 << ", " << b0 << ", " << g0 << ", " << ng0);
//    DCB_INIT(d1 << ", " << b1 << ", " << g1 << ", " << ng1);
    // (dimension, block dimension, grid location, 0, grid size)

//    DCB_INIT("l0,l1: " << l0 << ", " << l1 << ", " << setting.which);
//
//    data = new float[l0*l1];
//
//    lld = _max(l0, 1);
//
//    mint_t success = -1;
//    descinit(desc_local, 
//      &d0, &d1, 
//      &d0, &d1, 
//      &mint_zero, &mint_zero, 
//      &setting.handle, 
//      &d0, 
//      &success);
//    if(success != 0) {
//      throw std::runtime_error("descinit for desc_local failed");
//    }
//
//    success = -1;
//    descinit(desc, 
//      &d0, &d1,
//      &b0, &b1,
//      &mint_zero, &mint_zero, 
//      &setting.handle, 
//      &lld,
//      &success);
//    if(success != 0) {
//      throw std::runtime_error("descinit for desc (global) failed");
//    }
  }

  ~dist_matmul_t() { 
    if(data) {	  
      delete data;
    }
  }

  void init() {
    if(setting.which == 0) {
      throw std::runtime_error("don't call init() with processor zero");
    }
    auto& [d0,d1] = dims;
    psgeadd(
      &char_no_trans, 
      &d0, &d1, 
      &float_one, nullptr, &mint_one, &mint_one, desc_local, 
      &float_zero, data,   &mint_one, &mint_one, desc);
  }
  void init(float* global_data) {
    if(setting.which != 0) {
      throw std::runtime_error("don't call init(float*) with processor not-zero");
    }
    auto& [d0,d1] = dims;
    psgeadd(
      &char_no_trans, 
      &d0, &d1, 
      &float_one,  global_data, &mint_one, &mint_one, desc_local, 
      &float_zero, data,        &mint_one, &mint_one, desc);
  }

  void init(function<vector<float>()> f) {
    if(setting.which != 0) { 
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
    if(setting.which != 0) { 
      return init(); 
    }

    vector<float> data;
    auto [d0,d1] = dims;
    data.reserve(d0*d1);
    for(mint_t i = 0; i != d0; ++i) {
    for(mint_t j = 0; j != d1; ++j) {
      data.push_back(f(i,j));
    }}

    init(data.data());
  }
  
  void print() {
    auto const& [l0,l1] = local_dims;
    auto const& [g0,g1] = setting.grid_location;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "proc:        " << setting.which  << std::endl;
    std::cout << "grid:       [" << g0 << ", " << g1 << "]" << std::endl;
    std::cout << "local dims: [" << l0 << ", " << l1 << "]" << std::endl;
    for(mint_t i = 0; i != l0; ++i) {
    for(mint_t j = 0; j != l1; ++j) {
      std::cout << data[i*l1 + j] << " ";
    }}
    std::cout << std::endl;
  }

  float* data;

  setting_t const& setting;

  dims_t local_dims;
  dims_t dims;
  dims_t blocks;
  
  desc_t desc;
  desc_t desc_local;
  mint_t lld;
};

int main() {
  mint_t nB = 2;
  mint_t nD = 1024;

  dims_t grid_dims(nB,nB);
  setting_t setting(grid_dims);

  dims_t dims(nD,nD);
  dims_t blocks(nB,nB);

  dist_matmul_t A(setting, dims, blocks);
//
//  A.init([](int i, int j){ return 10*i + j; });
//
//  A.print();
}
