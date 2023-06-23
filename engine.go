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

var (
	_ tensor.Adder    = &Engine{}
	_ tensor.MatMuler = &Engine{}
)

const library = `
kernel void add(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}

kernel void addScalar(device const float* inVec,
                      device const float* inScalar,
                      device float* result,
                      uint index [[thread_position_in_grid]]) {
    result[index] = inVec[index] + *inScalar;
}

kernel void sub(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] - inB[index];
}

kernel void mul(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] * inB[index];
}
`

type byteslice []byte

func (b byteslice) Uintptr() uintptr { return uintptr(unsafe.Pointer(&b[0])) }
func (b byteslice) MemSize() uintptr { return uintptr(len(b)) }

type Engine struct {
	d *Device
	q CommandQueue
	l Library

	fns map[string]Function
}

func NewEngine(d *Device) *Engine {
	l, err := d.MakeLibrary(library)
	if err != nil {
		panic(err)
	}
	add, err := l.MakeFunction("add")
	if err != nil {
		panic(err)
	}
	return &Engine{
		d: d,
		q: MakeCommandQueue(d),

		fns: map[string]Function{"add": add},
	}
}

func (e *Engine) AllocAccessible() bool                   { return true }
func (e *Engine) Alloc(size int64) (tensor.Memory, error) { return AllocMBuf(e.d, size), nil } // TODO handle errors
func (e *Engine) Free(mem tensor.Memory, size int64) error {
	mBuf, ok := mem.(Buffer)
	if !ok {
		return errors.Errorf("Expected a Buffer. Got a Memory of %T instead", mem)
	}
	mBuf.Free()
	return nil
}
func (e *Engine) Memset(mem tensor.Memory, val interface{}) error     { panic("NYI") }
func (e *Engine) Memclr(mem tensor.Memory)                            { panic("NYI") }
func (e *Engine) Memcpy(dst, src tensor.Memory) error                 { panic("NYI") }
func (e *Engine) Accessible(mem tensor.Memory) (tensor.Memory, error) { panic("NYI") }
func (e *Engine) WorksWith(order tensor.DataOrder) bool               { return true } // for now

func (e *Engine) Add(a, b tensor.Tensor, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	reuse, safe, _, _, err := e.handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), opts...)
	if err != nil {
		return nil, errors.Wrap(err, "Add()")
	}
	cmdBuf := e.q.CommandBuffer()
	pso, err := e.d.MakeComputePipeline(e.fns["add"])
	if err != nil {
		return nil, err
	}
	elements := a.Shape().TotalSize()
	switch {
	case safe && reuse == nil:
		// make reuse
		reuseMem, err := e.Alloc(int64(elements * 4))
		if err != nil {
			return nil, errors.Wrapf(err, "Unable to allocate %d float32s for the result", elements)
		}
		reuse = tensor.New(tensor.WithShape(a.Shape().Clone()...), tensor.Of(a.Dtype()), tensor.WithEngine(e), tensor.FromMemory(reuseMem.Uintptr(), reuseMem.MemSize()))
		fallthrough
	case safe && reuse != nil:
		C.RunBinFunc(cmdBuf.b, pso.p, memAsMBuf(a).b, memAsMBuf(b).b, memAsMBuf(reuse).b, C.size_t(elements))
		retVal = reuse
		return
	case !safe:
		C.RunBinFunc(cmdBuf.b, pso.p, memAsMBuf(a).b, memAsMBuf(b).b, memAsMBuf(a).b, C.size_t(elements))
		retVal = a
		return
	}
	panic("Unreachable")
}

func (e *Engine) AddScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	// reuse, safe, _, _, err := e.handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), opts...)
	// if err != nil {
	// 	return nil, errors.Wrap(err, "AddScalar()")
	// }
	// cmdBuf := e.q.CommandBuffer()
	// pso, err := e.d.MakeComputePipeline(e.fns["addScalar"])
	// if err != nil {
	// 	return nil, err
	// }
	// elements := a.Shape().TotalSize()
	// switch {
	// case safe && reuse == nil:
	// 	// make reuse
	// 	reuseMem, err := e.Alloc(int64(elements * 4))
	// 	if err != nil {
	// 		return nil, errors.Wrapf(err, "Unable to allocate %d float32s for the result", elements)
	// 	}
	// 	reuse = tensor.New(tensor.WithShape(a.Shape().Clone()...), tensor.Of(a.Dtype()), tensor.WithEngine(e), tensor.FromMemory(reuseMem.Uintptr(), reuseMem.MemSize()))
	// 	fallthrough
	// case safe && reuse != nil:
	// 	if leftTensor {
	// 		C.RunBinFunc(cmdBuf.b, pso.p, memAsMBuf(a).b, memAsMBuf(b).b, memAsMBuf(reuse).b, C.size_t(elements))
	// 	} else {
	// 		C.RunBinFunc(cmdBuf.b, pso.p, memAsMBuf(b).b, memAsMBuf(a).b, memAsMBuf(reuse).b, C.size_t(elements))
	// 	}
	// 	retVal = reuse
	// 	return
	// case !safe:
	// 	if leftTensor {
	// 		C.RunBinFunc(cmdBuf.b, pso.p, memAsMBuf(a).b, memAsMBuf(b).b, memAsMBuf(a).b, C.size_t(elements))
	// 	} else {
	// 		C.RunBinFunc(cmdBuf.b, pso.p, memAsMBuf(b).b, memAsMBuf(a).b, memAsMBuf(a).b, C.size_t(elements))
	// 	}
	// 	retVal = a
	// 	return
	// }
	panic("Unreachable")
}

func (e *Engine) MatMul(a, b, prealloc tensor.Tensor) error {
	// select {
	// case <-ctx.Done():
	// 	return errors.New("NoOp")
	// default:
	// }
	ad, bd, retVal, err := e.checkValidMatmulInput(a, b, prealloc)
	if err != nil {
		return err
	}

	aDesc := pls(desc2MDesc(ad))
	bDesc := pls(desc2MDesc(bd))
	cDesc := pls(desc2MDesc(retVal))

	aBuf := memAsMBuf(ad)
	bBuf := memAsMBuf(bd)
	cBuf := memAsMBuf(retVal)

	A, B, C := NewMatrix(aBuf, aDesc), NewMatrix(bBuf, bDesc), NewMatrix(cBuf, cDesc)

	cmdBuf := e.q.CommandBuffer()
	return MPSMatMul(cmdBuf, A, B, C)
}

func (e *Engine) MatVecMul(a, b, prealloc tensor.Tensor) error {
	// select {
	// case <-ctx.Done():
	// 	return errors.New("NoOp")
	// default:
	// }
	ad, bd, retVal, err := e.checkValidMatVecMulInput(a, b, prealloc)
	if err != nil {
		return nil
	}
	aDesc := pls(desc2MDesc(ad))
	bDesc := pls(desc2VDesc(bd))
	cDesc := pls(desc2VDesc(retVal))

	aBuf := memAsMBuf(ad)
	bBuf := memAsMBuf(bd)
	cBuf := memAsMBuf(retVal)

	A, v, c := NewMatrix(aBuf, aDesc), NewVector(bBuf, bDesc), NewVector(cBuf, cDesc)
	cmdBuf := e.q.CommandBuffer()
	return MPSMatVecMul(cmdBuf, A, v, c)
}

func (e *Engine) checkValidDtype(ts ...tensor.Tensor) error {
	for i, t := range ts {
		if t.Dtype() != tensor.Float32 {
			return errors.Errorf("Expected all inputs to be Float32s. Input %d is of type %v", i, t.Dtype())
		}
	}
	return nil
}

func (e *Engine) LogSoftMax(x tensor.Tensor, axis int, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	panic("NYI")
}
func (e *Engine) LogSoftMaxB(output, grad tensor.Tensor, axis int, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	panic("NYI")
}
func (e *Engine) SoftMax(x tensor.Tensor, axis int, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	reuse, safe, _, _, err := e.handleFuncOpts(x.Shape(), x.Dtype(), x.DataOrder(), opts...)
	if err != nil {
		return nil, err
	}

	elements := x.Shape().TotalSize()
	switch {
	case safe && reuse == nil:
		// then we need to make reuse
		reuseMem, err := e.Alloc(int64(elements * 4))
		if err != nil {
			return nil, errors.Wrapf(err, "Unable to allocate %d float32s for the resulting tensor", elements)
		}
		reuse = tensor.New(tensor.WithShape(x.Shape().Clone()...), tensor.Of(x.Dtype()), tensor.WithEngine(e), tensor.FromMemory(reuseMem.Uintptr(), reuseMem.MemSize()))
		fallthrough
	case safe && reuse != nil:
		// we can just reuse reuse
		xd := x.(tensor.DenseTensor)
		xDesc := pls(desc2MDesc(xd))
		rDesc := pls(desc2MDesc(reuse))
		xBuf := memAsMBuf(xd)
		rBuf := memAsMBuf(reuse)
		A, Out := NewMatrix(xBuf, xDesc), NewMatrix(rBuf, rDesc)

		cmdBuf := e.q.CommandBuffer()
		err = MPSSoftmax(cmdBuf, A, Out)
		return reuse, err
	case !safe:
		// then A is the result as well as input

	}
	panic("Unreachable")
}
func (e *Engine) SoftMaxB(output, grad tensor.Tensor, axis int, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	panic("NYI")
}

func (e *Engine) checkValidMatmulInput(a, b, ret tensor.Tensor) (ad, bd, retVal tensor.DenseTensor, err error) {
	if a.Dtype() != tensor.Float32 || a.Dtype() != b.Dtype() || b.Dtype() != ret.Dtype() {
		return nil, nil, nil, errors.New("Expected a and b and retVal all to have the same Dtype")
	}
	ad, ok := a.(tensor.DenseTensor)
	if !ok {
		return nil, nil, nil, errors.New("Expected a to be a DenseTensor")
	}
	bd, ok = b.(tensor.DenseTensor)
	if !ok {
		return nil, nil, nil, errors.New("Expected b to be a DenseTensor")
	}
	retVal, ok = ret.(tensor.DenseTensor)
	if !ok {
		return nil, nil, nil, errors.New("Expected retVal to be a DenseTensor")
	}
	if ad.Shape().Dims() != 2 || bd.Shape().Dims() != 2 || retVal.Shape().Dims() != 2 {
		return nil, nil, nil, errors.New("Expected a, b, and retVal to be 2D")
	}
	if ad.Shape()[1] != bd.Shape()[0] {
		return nil, nil, nil, errors.New("Expected the inner dimensions of a and b to match")
	}

	return ad, bd, retVal, nil
}

func (e *Engine) checkValidMatVecMulInput(a, b, ret tensor.Tensor) (ad, bd, retVal tensor.DenseTensor, err error) {
	if a.Dtype() != tensor.Float32 || a.Dtype() != b.Dtype() || b.Dtype() != ret.Dtype() {
		return nil, nil, nil, errors.New("Expected a and b and retVal all to have the same Dtype")
	}
	ad, ok := a.(tensor.DenseTensor)
	if !ok {
		return nil, nil, nil, errors.New("Expected a to be a DenseTensor")
	}
	bd, ok = b.(tensor.DenseTensor)
	if !ok {
		return nil, nil, nil, errors.New("Expected b to be a DenseTensor")
	}
	retVal, ok = ret.(tensor.DenseTensor)
	if !ok {
		return nil, nil, nil, errors.New("Expected retVal to be a DenseTensor")
	}
	if ad.Shape().Dims() != 2 || bd.Shape().IsVectorLike() || retVal.Shape().IsVectorLike() {
		return nil, nil, nil, errors.New("Expected a to be 2D and b and retVal to be  vectorlike")
	}
	if ad.Shape()[1] != bd.Shape()[0] {
		return nil, nil, nil, errors.New("Expected the inner dimensions of a and b to match")
	}

	return ad, bd, retVal, nil
}

func (e *Engine) handleFuncOpts(expectedShape tensor.Shape, expectedType tensor.Dtype, do tensor.DataOrder, opts ...tensor.FuncOpt) (reuse tensor.DenseTensor, safe, toReuse, incr bool, err error) {
	fo := tensor.ParseFuncOpts(opts...)
	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	toReuse = reuseT != nil

	if toReuse {
		var ok bool
		if reuse, ok = reuseT.(tensor.DenseTensor); !ok {
			err = errors.Errorf("Expected a DenseTensor in Reuse. Got %T instead", reuseT)
			return
		}

		if reuse.Dtype() != expectedType {
			err = errors.Errorf("Expected %v. Got %v instead", expectedType, reuse.Dtype())
			return
		}
		if reuse.Shape().TotalSize() != expectedShape.TotalSize() {
			err = errors.Errorf("Expecte shape to be equivalent to %v. Got %v instead", expectedShape, reuse.Shape())
			return
		}
		if err = reuse.Reshape(expectedShape.Clone()...); err != nil {
			return
		}
	}
	return
}
