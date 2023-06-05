#include <math.h>
#include <mkl_scalapack.h>
#include <time.h>

#define MAX(x, y) ((x)>(y) ? (x):(y))

typedef MKL_INT MDESC[ 9 ];
const double  zero = 0.0E+0, one = 1.0E+0, two = 2.0E+0, negone = -1.0E+0;
const MKL_INT i_zero = 0, i_one = 1, i_four = 4, i_negone = -1;
const char    trans = 'N';

// Row major
// res = a*b, where a is n*k matrix, b is k*m matrix
void distributed_dgemm(double *res, double *a, double *b, int n, int k, int m)
{
}

void print(double *a, int n, int m)
{
/*	int i, j;
	for (i=0; i<n; ++i)
	{
			for (j=0; j<m; ++j)
					printf("%lf\t", a[i+j*n]);

			printf("\n");
	}*/
}

double* gen_a(int n, int m)
{
		double *a = (double*)calloc(n*m, sizeof(double));
		int i, j;
		for (i=0; i<n; ++i)
				for (j=0; j<m; ++j)
						a[i+j*n] = i*10 + j;
		return a;
}

double* gen_b(int n, int m)
{
		double *b = (double*)calloc(n*m, sizeof(double));
		int i, j;
		for (i=0; i<n; ++i)
				for (j=0; j<m; ++j)
						b[i+j*n] = i*j;
		return b;
}

int main()
{
	const MKL_INT m = 1000;
	const MKL_INT k = 100000;
	const MKL_INT n = 1000;
	const MKL_INT nb = 100;
	const MKL_INT nprow = 2;
	const MKL_INT npcol = 2;

    MKL_INT iam, nprocs, ictxt, myrow, mycol;
    MDESC   descA, descB, descC, descA_local, descB_local, descC_local;
	MKL_INT info;
	MKL_INT a_m_local, a_n_local, b_m_local, b_n_local, c_m_local, c_n_local;
	MKL_INT a_lld, b_lld, c_lld;

    blacs_pinfo_( &iam, &nprocs );
    blacs_get_( &i_negone, &i_zero, &ictxt );
    blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

    double *a = 0;
    double *b = 0;
	double *c = 0;

    if (iam==0)
    {
		a = gen_a(m, k);
		b = gen_b(k, n);
		c = (double*)calloc(m*n, sizeof(double));

		puts("a=");
		print(a, m, k);

		puts("b=");
		print(b, k, n);
    }

    //    a       b       c
    // m  row     row     row
    // n  col     col     col
    a_m_local = numroc_( &m, &nb, &myrow, &i_zero, &nprow );
    a_n_local = numroc_( &k, &nb, &mycol, &i_zero, &npcol );

	b_m_local = numroc_( &k, &nb, &myrow, &i_zero, &nprow );
	b_n_local = numroc_( &n, &nb, &mycol, &i_zero, &npcol );

    c_m_local = numroc_( &m, &nb, &myrow, &i_zero, &nprow );
	c_n_local = numroc_( &n, &nb, &mycol, &i_zero, &npcol );

    double *A = (double*) calloc( a_m_local * a_n_local, sizeof( double ) );
    double *B = (double*) calloc( b_m_local * b_n_local, sizeof( double ) );
    double *C = (double*) calloc( c_m_local * c_n_local, sizeof( double ) );

    a_lld = MAX( a_m_local, 1 );
	b_lld = MAX( b_m_local, 1 );
	c_lld = MAX( c_m_local, 1 );

	if (iam==0)
	{
			printf("a_m_local = %d\ta_n_local = %d\tb_m_local = %d\tb_n_local = %d\tc_m_local = %d\tc_n_local = %d\n", a_m_local, a_n_local, b_m_local, b_n_local,
							c_m_local, c_n_local);
			printf("a_lld = %d\tb_lld = %d\tc_lld = %d\n", a_lld, b_lld, c_lld);
	}

    descinit_( descA_local, &m, &k, &m, &k, &i_zero, &i_zero, &ictxt, &m, &info );
    descinit_( descB_local, &k, &n, &k, &n, &i_zero, &i_zero, &ictxt, &k, &info );
    descinit_( descC_local, &m, &n, &m, &n, &i_zero, &i_zero, &ictxt, &m, &info );

    descinit_( descA, &m, &k, &nb, &nb, &i_zero, &i_zero, &ictxt, &a_lld, &info );
    descinit_( descB, &k, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &b_lld, &info );
    descinit_( descC, &m, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &c_lld, &info );

	printf("Rank %d: start distribute data\n", iam);
    pdgeadd_( &trans, &m, &k, &one, a, &i_one, &i_one, descA_local, &zero, A, &i_one, &i_one, descA );
    pdgeadd_( &trans, &k, &n, &one, b, &i_one, &i_one, descB_local, &zero, B, &i_one, &i_one, descB );
	printf("Rank %d: finished distribute data\n", iam);

	if (iam==0)
	{
			puts("a");
			print(A, a_m_local, a_n_local);
			puts("b");
			print(B, b_m_local, b_n_local);
	}

    pdgemm_( "N", "N", &m, &n, &k, &one, A, &i_one, &i_one, descA, B, &i_one, &i_one, descB,
             &zero, C, &i_one, &i_one, descC );
	printf("Rank %d: finished dgemm\n", iam);
	if (iam == 0)
	{
			puts("c");
			print(C, c_m_local, c_n_local);
	}

	pdgeadd_( &trans, &m, &n, &one, C, &i_one, &i_one, descC, &zero, c, &i_one, &i_one, descC_local);

	if (iam==0)
	{
			puts("global c");
			print(c, m, n);
	}

	free(A);
	free(B);
	free(C);
	if (iam==0)
	{
			free(a);
			free(b);
			free(c);
	}

    blacs_gridexit_( &ictxt );
    blacs_exit_( &i_zero );
}

