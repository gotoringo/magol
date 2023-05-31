// +build darwin

#import <Foundation/Foundation.h>
#import <MetalKit/MetalKit.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "magol.h"
#include <stdio.h>
#include <stdlib.h>

struct Device CreateSystemDefaultDevice() {
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	if (!device) {
		struct Device d;
		d.Device = NULL;
		return d;
	}

	struct Device d;
	d.Device = device;
	d.IsHeadless = device.headless;
	d.IsLowPower = device.lowPower;
	d.IsRemovable = device.removable;
	d.RegistryID = device.registryID;
	d.Name = device.name.UTF8String;
	return d;
}

void* MakeCommandQueue(void* device){
	return [(id<MTLDevice>)device newCommandQueue];
}

void* MakeCommandBuffer(void* cmdq){
	return [(id<MTLCommandQueue>)cmdq commandBuffer];
}

// https://developer.apple.com/documentation/metal/mtldevice/1433429-makebuffer?language=objc
void* Buf2MBuf(void* device, const void* bytes, size_t memsize) {
	return [(id<MTLDevice>)device newBufferWithBytes:(const void*)bytes
						length:(NSUInteger)memsize
					       options:MTLResourceStorageModeShared]; // TODO - abstract this to Go
}

void* AllocMBuf(void* device, size_t memsize){
	return [(id<MTLDevice>)device newBufferWithLength:(NSUInteger)memsize
		                                 options: MTLResourceStorageModeShared];
}

void* FreeMBuf(void* metalBuf){
	id<MTLBuffer> mBuf  = ((id<MTLBuffer>)metalBuf);
	[mBuf setPurgeableState:MTLPurgeableStateEmpty];
	[mBuf release]; // note, release doesn't actually dealloc. It just reduces the reference count
}

// https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixdescriptor/2873331-matrixdescriptorwithrows?language=objc
void* MatrixDesc(uint_t rows, uint_t cols, uint_t rowBytes){
	return [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows
						 columns:(NSUInteger)cols
					        rowBytes:(NSUInteger)rowBytes
						dataType:MPSDataTypeFloat32]; // TODO abstract data types
}

//https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix/2143201-initwithbuffer?language=objc
void* Matrix(void* buf, void* desc){
	return [[MPSMatrix alloc] initWithBuffer:(id<MTLBuffer>)buf
				      descriptor:(MPSMatrixDescriptor*)desc];
}

void* MBuf2Buf(void* dst,  void* metalbuf, size_t len) {
	id<MTLBuffer> mbuf  = (id<MTLBuffer>)metalbuf;
	memcpy(dst, [mbuf contents], len);
}


void CmdBuf_Enqueue(void* cmdBuf) {
	[(id<MTLCommandBuffer>)cmdBuf enqueue];
}

void* MakeComputeCommandEncoder(void* cmdbuf) {
	id<MTLComputeCommandEncoder> computeEncoder = [(id<MTLCommandBuffer>)cmdbuf computeCommandEncoder];
	return computeEncoder;
}

void RunBinFunc(void* commandbuffer, void* pipelineFunc, void* bufA, void* bufB, void* bufC, size_t arrlen) {
	id<MTLCommandBuffer> cmdbuf = (id<MTLCommandBuffer>)commandbuffer;
	id<MTLComputePipelineState> pso = (id<MTLComputePipelineState>)pipelineFunc;
	id<MTLComputeCommandEncoder> computeEncoder = [cmdbuf computeCommandEncoder];
	[computeEncoder setComputePipelineState:pso];
	[computeEncoder setBuffer:(id<MTLBuffer>)bufA offset:0 atIndex:0];
	[computeEncoder setBuffer:(id<MTLBuffer>)bufB offset:0 atIndex:1];
	[computeEncoder setBuffer:(id<MTLBuffer>)bufC offset:0 atIndex:2];

	NSUInteger len = (NSUInteger)arrlen;
	MTLSize gridSize = MTLSizeMake(len, 1, 1);
	NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
	if (threadGroupSize > arrlen) {
		threadGroupSize = arrlen;
	}
	MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

	// encode the command to execute threads
	[computeEncoder dispatchThreads:gridSize
		  threadsPerThreadgroup:threadgroupSize];

	// end compute pass
	[computeEncoder endEncoding];

	[cmdbuf commit];

	[cmdbuf waitUntilCompleted];
}

Res_t MakeLibrary(void* device, const char* src, size_t len) {
	NSError* error;
	id<MTLLibrary> lib = [(id<MTLDevice>)device newLibraryWithSource: [[NSString alloc]  initWithBytes:src length:len encoding:NSUTF8StringEncoding]
								 options: NULL
								   error:&error];
	Res_t l;
	l.Ptr = lib;
	if (!lib) {
		l.Err = error.localizedDescription.UTF8String;
	}
	return l;
}

void* MakeFunction(void* lib, const char* name){ return [(id<MTLLibrary>)lib newFunctionWithName:[NSString stringWithUTF8String:name]]; }

Res_t MakeComputePipeline(void* device, void* function) {
	NSError* error;
	id<MTLComputePipelineState> cp = [(id<MTLDevice>)device newComputePipelineStateWithFunction: (id<MTLFunction>)function
											      error:&error];
	Res_t retVal;
	retVal.Ptr = cp;
	if (!cp) {
		retVal.Err = error.localizedDescription.UTF8String;
	}
	return retVal;
}


/* LINALG */

void* matmul(void* commandBuffer, void* matrixA, void* matrixB, void* matrixC, bool transA, bool transB) {
	id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)commandBuffer;
	MPSMatrix *A = (MPSMatrix*)matrixA;
	MPSMatrix *B = (MPSMatrix*)matrixB;
	MPSMatrix *C = (MPSMatrix*)matrixC;
    // Create a MPSMatrixMultiplication kernel
    MPSMatrixMultiplication* matrixMultiplication = [[MPSMatrixMultiplication alloc] initWithDevice:cmdBuf.device
										      transposeLeft:transA // TODO abstract
										     transposeRight:transB // TODO abstract
											 resultRows:A.rows
										      resultColumns:B.columns
										    interiorColumns:A.columns
											      alpha:1.0 // TODO abstract
											       beta:0.0]; // TODO abstract


    // Perform matrix multiplication
    [matrixMultiplication encodeToCommandBuffer:cmdBuf
				     leftMatrix:A
				    rightMatrix:B
				   resultMatrix:C];

    [cmdBuf commit]; // TODO THIS IS A BAD IDEA. MOVE CONTROL STRUCTURES OUT
    [cmdBuf waitUntilCompleted];
}

/* NN */
void* softmax(void* commandBuffer, void* mat, void* out) {
	id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>)commandBuffer;
	MPSMatrix *A = (MPSMatrix*)mat;
	MPSMatrix *B = (MPSMatrix*)out;

	// Create a MPSMatrixSoftMax kernel
        MPSMatrixSoftMax* softmax = [[MPSMatrixSoftMax alloc] initWithDevice:cmdBuf.device];

	// Perform softmax
	[softmax encodeToCommandBuffer:cmdBuf
			   inputMatrix:A
			  resultMatrix:B];

	[cmdBuf commit]; // TODO THIS IS A BAD IDEA. MOVE CONTROL STRUCTURES OUT
	[cmdBuf waitUntilCompleted];
}

/* TMP UTILITIES */
void printMatrix(void* in) {
	MPSMatrix* matrixC = (MPSMatrix*)in;
    float *rawPointer = (float *)matrixC.data.contents;
    NSUInteger count = matrixC.rows * matrixC.columns;
    float *typedPointer = (float *)rawPointer;
    for (NSUInteger i = 0; i < count; i++) {
        printf("%f\n", typedPointer[i]);
    }
}
