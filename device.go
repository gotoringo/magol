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

// Device is a representation of a GPU
//
// See: https://developer.apple.com/documentation/metal/mtldevice?language=objc.
type Device struct {
	d unsafe.Pointer

	isHeadless  bool
	isLowPower  bool
	isRemovable bool
	registryID  uint64
	name        string
}

func NewDevice() *Device {
	d := C.CreateSystemDefaultDevice()
	if d.Device == nil {
		return nil
	}
	return &Device{
		d: d.Device,

		isHeadless:  bool(d.IsHeadless),
		isLowPower:  bool(d.IsLowPower),
		isRemovable: bool(d.IsRemovable),
		registryID:  uint64(d.RegistryID),
		name:        C.GoString(d.Name),
	}
}

func (d *Device) MakeLibrary(src string) (Library, error) {
	l := C.MakeLibrary(d.d, C.CString(src), C.size_t(len(src)))
	if l.Ptr == nil {
		return Library{}, errors.New(C.GoString(l.Err))
	}
	return Library{l.Ptr}, nil
}

func (d *Device) MakeComputePipeline(fn Function) (ComputePipeline, error) {
	cp := C.MakeComputePipeline(d.d, fn.f)
	if cp.Ptr == nil {
		return ComputePipeline{}, errors.New(C.GoString(cp.Err))
	}
	return ComputePipeline{cp.Ptr}, nil
}
