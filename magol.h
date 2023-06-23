// +build darwin

typedef unsigned long uint_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned long long uint64_t;

typedef signed long int_t;

struct Device {
	void *       Device;
	bool         IsHeadless;
	bool         IsLowPower;
	bool         IsRemovable;
	uint64_t     RegistryID;
	const char * Name;
};
struct Device CreateSystemDefaultDevice();
void* MakeCommandQueue(void* device);
void* MakeCommandBuffer(void* cmdq);
void* Buf2MBuf(void* device, const void* bytes, size_t memsize);
void* AllocMBuf(void* device, size_t memsize);
void* FreeMBuf(void* mBuf);
void* MatrixDesc(uint_t rows, uint_t cols, uint_t rowBytes);
void* Matrix(void* buf, void* desc);
void* VectorDesc(uint_t length);
void* Vector(void* buf, void* desc);
void* MBuf2Buf(void* dst,  void* metalbuf, size_t len);
void* MakeComputeCommandEncoder(void* cmdbuf);
void CmdBuf_Enqueue(void* cmdBuf);
void RunBinFunc(void* commandbuffer, void* pipelineFunc, void* bufA, void* bufB, void* bufC, size_t arrlen);
typedef struct Res {
	void* Ptr; // the actual pointer to the object (library, function, computepipeline, etc)
	const char* Err;
} Res_t;

Res_t MakeLibrary(void* device, const char* src, size_t len);
void* MakeFunction(void* lib, const char* name);
Res_t MakeComputePipeline(void* device, void* function);

/* Linalg */
void* matmul(void* commandBuffer, void* matrixA, void* matrixB, void* matrixC, bool transA, bool transB);
void* matvecmul(void* commandBuffer, void* matrixA, void* vecB, void* vecC, bool transMat);
/* NN */
void* softmax(void* commandBuffer, void* mat, void* out);
// TMP

void printMatrix(void* in);
