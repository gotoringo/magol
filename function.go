package magol

/*
#cgo LDFLAGS: -framework Metal -framework CoreGraphics -framework Foundation -framework MetalPerformanceShaders
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "magol.h"
*/
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

// Function represents a function that is executed by the GPU.
//
// See: https://developer.apple.com/documentation/metal/mtlfunction?lang=objc
type Function struct{ f unsafe.Pointer }

// Library is a collection of functions.
//
// See: https://developer.apple.com/documentation/metal/mtllibrary?lang=objc
type Library struct{ l unsafe.Pointer }

func (l Library) MakeFunction(name string) (Function, error) {
	f := C.MakeFunction(l.l, C.CString(name))
	if f == nil {
		return Function{}, errors.Errorf("Function %v not found", name)
	}
	return Function{f}, nil
}

// ComputePipeline represents a compute pipeline.
//
// See: https://developer.apple.com/documentation/metal/mtlcomputepipelinestate?language=objc
type ComputePipeline struct {
	p unsafe.Pointer
}
