#include <cmath>
#include <cstring>
#include <cblas.h>

/* multiply (row stored) a[m*k] with trans(b[n*k]) to get c[m*n],
 * and add the vector bias for each column */
void cpu_gemm(int m, int n, int k, float *a, float *b, float *c, float *bias)
{
	for (int i = 0; i < m; i++) {
		float *p = &c[i*n];
		memcpy(p, bias, sizeof(float)*n);
	}
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, a, k, b, k, 1, c, n);
}

float relu(float a) {
  if (a < 0)
    return 0;
  return a;
}

float sigmoid(float a) { return 1.0 / (1.0 + exp(-a)); }

void cpu_activation(int type, size_t length, float *a, float *b) {
  if (type == 0) {
    for (int i = 0; i < length; i++) {
      (*(b + i)) = relu(*(a + i));
    }
  } else {
    for (int i = 0; i < length; i++) {
      (*(b + i)) = sigmoid(*(a + i));
    }
  }
}
