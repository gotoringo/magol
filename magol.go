//go:build darwin
// +build darwin

package magol

import (
	"unsafe"

	"gorgonia.org/tensor"
)

/*
#cgo LDFLAGS: -framework Metal -framework CoreGraphics -framework Foundation -framework MetalPerformanceShaders
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "magol.h"
*/
import "C"

type Dataer interface {
	Float32s() []float32
}

// Buffer is a memory slice in the GPU.
//
// See: https://developer.apple.com/documentation/metal/mtlbuffer?language=objc
type Buffer struct {
	b  unsafe.Pointer
	sz uintptr
}

func (b Buffer) Uintptr() uintptr { return uintptr(b.b) }
func (b Buffer) MemSize() uintptr { return b.sz }
func (b Buffer) Free()            { C.FreeMBuf(b.b); b.b = nil }

func AllocMBuf(device *Device, sz int64) Buffer {
	return Buffer{b: C.AllocMBuf(device.d, C.size_t(sz)), sz: uintptr(sz)}
}

func memAsMBuf(a tensor.Memory) Buffer {
	return Buffer{b: unsafe.Pointer(a.Uintptr()), sz: a.MemSize()}
}

func buf2MBuf(device *Device, data tensor.Memory) Buffer {
	bytes := unsafe.Pointer(data.Uintptr())
	len := int(data.MemSize())
	return Buffer{b: C.Buf2MBuf(device.d, bytes, C.size_t(len)), sz: data.MemSize()}
}

func Mbuf2Buf(dst tensor.Memory, src tensor.Memory) {
	switch src := src.(type) {
	case Buffer:
		C.MBuf2Buf(unsafe.Pointer(dst.Uintptr()), src.b, C.size_t(dst.MemSize()))
	default:
		C.MBuf2Buf(unsafe.Pointer(dst.Uintptr()), unsafe.Pointer(src.Uintptr()), C.size_t(dst.MemSize()))
	}

}

func GoSliceAsMBuf[T any](d *Device, s []T) Buffer {
	ptr := unsafe.Pointer(&s[0])
	var v T
	l := uintptr(len(s)) * unsafe.Sizeof(v)
	return Buffer{b: C.Buf2MBuf(d.d, ptr, C.size_t(l)), sz: l}
}

func debug(m *Matrix) {
	C.printMatrix(m.m)
}

/* UTILS */

func pls[T any](a T, err error) T {
	if err != nil {
		panic(err)
	}
	return a
}
