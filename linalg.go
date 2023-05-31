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
	"gorgonia.org/tensor"
)

// MatrixDesc is a matrix descriptor, much like tensor's AP.
//
// See: https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixdescriptor?language=objc
type MatrixDesc struct {
	d unsafe.Pointer
}

func (d MatrixDesc) isDesc() {}

func desc2MDesc(t tensor.DenseTensor) (MatrixDesc, error) {
	desc := t.Info()
	if desc.Dims() != 2 {
		return MatrixDesc{}, errors.New("Expected Matrix")
	}
	shp := desc.Shape()
	rows := shp[0]
	cols := shp[1]
	rowBytes := int(t.Dtype().Size()) * cols
	mdesc := C.MatrixDesc(C.uint_t(rows), C.uint_t(cols), C.uint_t(rowBytes)) // TODO: type mapping
	if mdesc == nil {
		return MatrixDesc{}, errors.New("Failed to create Matrix Descriptor")
	}
	return MatrixDesc{mdesc}, nil
}

// Matrix represents a matrix.
// See: https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix?language=objc
type Matrix struct {
	m unsafe.Pointer
	b Buffer

	trans bool // whether the matrix is transposed or not
}

func NewMatrix(buf Buffer, desc MatrixDesc) *Matrix {
	return &Matrix{m: C.Matrix(buf.b, desc.d), b: buf}
}

func (m *Matrix) Buffer() Buffer { return m.b }

func (m *Matrix) ToggleTranspose() { m.trans = !m.trans }

// VectorDescriptor is a descriptor for vectors, much like tensor's AP.
//
// See: https://developer.apple.com/documentation/metalperformanceshaders/mpsvectordescriptor?language=objc
type VectorDescriptor struct {
	d unsafe.Pointer
}

func (d VectorDescriptor) isDesc() {}

// Vector represents a vector.
// See: https://developer.apple.com/documentation/metalperformanceshaders/mpsvector?language=objc
type Vector struct {
	v unsafe.Pointer
	b Buffer
}

func MPSMatMul(cmdBuf CommandBuffer, A, B, CM *Matrix) error {
	C.matmul(cmdBuf.b, A.m, B.m, CM.m, C.bool(A.trans), C.bool(B.trans))
	return nil
}

func MPSMatVecMul(cmdBuf CommandBuffer, M *Matrix, v *Vector, retVal *Vector) error { panic("NYI") }
