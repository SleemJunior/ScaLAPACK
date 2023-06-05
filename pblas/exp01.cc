#include <mkl_scalapack.h>

#include <iostream>
#include <vector>
#include <functional>

using std::vector;
using std::function;

const float zero = 0.0;
const float one  = 0.0;

typedef MKL_INT MDESC[ 9 ];

vector<float> 
generate_matrix(int nI, int nJ, function<float(int,int)> gen)
{
  vector<float> ret;
  ret.resize(nI*nJ);
  for (int i=0; i!=nI; ++i) {
  for (int j=0; j!=nJ; ++j) {
    ret[i*nJ+j] = gen(i,j);
  }}
  return ret; 
}

int main() {
  MKL_INT nprow = 1;
  MKL_INT npcol = 1;

  MKL_INT iam, nprocs, ictxt, myrow, mycol;
  MDESC   descA, descB, descC;
  MKL_INT info;
  MKL_INT a_m_local, a_n_local, b_m_local, b_n_local, c_m_local, c_n_local;
  MKL_INT a_lld, b_lld, c_lld;

  blacs_pinfo_( &iam, &nprocs );
  blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
  blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

}
