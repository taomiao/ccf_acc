#include <cmath>
/* multiply (row stored) a[m*k] with trans(b[n*k]) to get c[m*n],
 * and add the vector bias for each column */
void cpu_gemm(int m, int n, int k, float *a, float *b, float *c, float *bias)
{            
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			float sum=0;
			for(int w=0;w<k;w++){
				sum += a[i*k + w] * b[j*k+w];
			}
			sum += bias[j];
			c[i*n+j] = sum;
		}
	}
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
