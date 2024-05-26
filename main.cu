#include <stdexcept>
#include <device_launch_parameters.h>


const size_t WARP_SIZE = 32;


void full(float** mat, float val, size_t size)
{
  *mat = (float*)malloc(size * sizeof(float));

  for (size_t i = 0; i < size; i++)
	(*mat)[i] = val;
}

__global__ void simpleMatMul(float* a, float* b, float* c, size_t l, size_t m, size_t n)
{
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= l || col >= n)
	return;

  float sum = 0;
  for (size_t i = 0; i < m; i++) {
	sum += a[row * m + i] * b[i * n + col];
  }

  c[row * n + col] = sum;
}

__global__ void sharedMatMul(float* a, float* b, float* c, size_t l, size_t m, size_t n)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t tileCol = threadIdx.x;
  size_t tileRow = threadIdx.y;

  __shared__ float aTile[WARP_SIZE][WARP_SIZE];
  __shared__ float bTile[WARP_SIZE][WARP_SIZE];

  float sum = 0.f;
  bool isOutOfC = row >= l || col >= n;

  if (isOutOfC) {
	return;
  }

  for (size_t tileId = 0; tileId < (m - 1) / WARP_SIZE + 1; tileId++)
  {
	aTile[tileRow][tileCol] = a[row * m + (tileId * WARP_SIZE + tileCol)];
	bTile[tileRow][tileCol] = b[(tileId * WARP_SIZE + tileRow) * n + col];
	__syncthreads();

	for (size_t i = 0; i < WARP_SIZE; i++)
	  sum += aTile[tileRow][i] * bTile[i][tileCol];
	__syncthreads();
  }

  c[row * n + col] = sum;
}

__global__ void warpIntrinsicsMatMul(float* a, float* b, float* c, size_t l, size_t m, size_t n)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t tileCol = threadIdx.x;
  size_t tileRow = threadIdx.y;

  __shared__ float aTile[WARP_SIZE][WARP_SIZE];
  __shared__ float bTile[WARP_SIZE][WARP_SIZE + 1]; // + 1 to avoid bank conflicts

  float cVal = 0.f;
  bool isOutOfC = row >= l || col >= n;

  if (isOutOfC) {
	return;
  }

  for (size_t tileId = 0; tileId < (m - 1) / WARP_SIZE + 1; tileId++)
  {
	aTile[tileRow][tileCol] = !isOutOfC ? a[row * m + (tileId * WARP_SIZE + tileCol)] : 0.f;
	bTile[tileRow][tileCol] = !isOutOfC ? b[(tileId * WARP_SIZE + tileRow) * n + col] : 0.f;
	__syncthreads();

	float aTileLocal = aTile[tileRow][tileCol];
	__syncwarp();
	for (size_t i = 0; i < WARP_SIZE; i++)
	  cVal += __shfl_sync(0xffffffff, aTileLocal, i) * bTile[i][tileCol];
	__syncthreads();
  }

  c[row * n + col] = cVal;
}

int main()
{
  cudaError_t cudaStatus;
  size_t l = 1 << 4, m = 1 << 4, n = 1 << 4;

  float* a, * b, * c;
  float* a_dev, * b_dev, * c_dev;

  full(&a, 1.f, l * m);
  full(&b, 1.f, m * n);
  c = (float*)malloc(l * n * sizeof(float));

  cudaStatus = cudaMalloc(&a_dev, l * m * sizeof(float));
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed! [%d]", int(cudaStatus));
	return 1;
  }

  cudaStatus = cudaMalloc(&b_dev, m * n * sizeof(float));
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed! [%d]", int(cudaStatus));
	return 1;
  }

  cudaStatus = cudaMalloc(&c_dev, l * n * sizeof(float));
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed! [%d]", int(cudaStatus));
	return 1;
  }

  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed? [%d]", int(cudaStatus));
	return 1;
  }

  cudaStatus = cudaMemcpy(a_dev, a, l * m * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed! [%d]", int(cudaStatus));
	return 1;
  }

  cudaStatus = cudaMemcpy(b_dev, b, m * n * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed! [%d]", int(cudaStatus));
	return 1;
  }

  dim3 blockInGrid((n - 1ULL) / WARP_SIZE + 1ULL, (l - 1ULL) / WARP_SIZE + 1ULL);
  dim3 threadInBlock(WARP_SIZE, WARP_SIZE);

  int mode;
  scanf("%d", &mode);

  switch (mode) {
  case 1:
	printf("simpleMatMul\n");
	simpleMatMul <<< blockInGrid, threadInBlock >>> (a_dev, b_dev, c_dev, l, m, n);
	break;
  case 2:
	printf("sharedMatMul\n");
	sharedMatMul <<< blockInGrid, threadInBlock >>> (a_dev, b_dev, c_dev, l, m, n);
	break;
  case 3:
	printf("warpIntrinsicsMatMul\n");
	warpIntrinsicsMatMul <<< blockInGrid, threadInBlock >>> (a_dev, b_dev, c_dev, l, m, n);
	break;
  default:
    printf("none was selected\n");
	return 1;
  }

  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceSynchronize failed! [%d]", int(cudaStatus));
	return 1;
  }

  cudaStatus = cudaMemcpy(c, c_dev, l * n * sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed! [%d]", int(cudaStatus));
	return 1;
  }

  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "last cudaStatus is not success: %d", int(cudaStatus));
	return 1;
  }

  printf("%f\n", c[0]);
  for (int i=0; i < l; i++) {
	for (int j=0; j < n; j ++) {
		printf("%f ", c[i*n + j]);
	}
	printf("\n");
  }

  free(a);
  free(b);
  cudaFree(a_dev);
  cudaFree(b_dev);
  free(c);
  cudaFree(c_dev);
}

