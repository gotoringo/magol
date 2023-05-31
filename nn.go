package magol

/*
#cgo LDFLAGS: -framework Metal -framework CoreGraphics -framework Foundation -framework MetalPerformanceShaders
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "magol.h"
*/
import "C"

func MPSSoftmax(cmdBuf CommandBuffer, A, retVal *Matrix) error {
	C.softmax(cmdBuf.b, A.m, retVal.m)
	return nil
}
