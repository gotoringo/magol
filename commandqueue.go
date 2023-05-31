package magol

import "unsafe"

/*
#cgo LDFLAGS: -framework Metal -framework CoreGraphics -framework Foundation
#include <stdlib.h>
#include <stdbool.h>
#include "magol.h"
*/
import "C"

type CommandQueue struct {
	q unsafe.Pointer
}

type CommandBuffer struct {
	b unsafe.Pointer
}

func MakeCommandQueue(dev *Device) CommandQueue     { return CommandQueue{C.MakeCommandQueue(dev.d)} }
func (q CommandQueue) CommandBuffer() CommandBuffer { return CommandBuffer{C.MakeCommandBuffer(q.q)} }
func (b CommandBuffer) Enqueue()                    { C.CmdBuf_Enqueue(b.b) }
func (b CommandBuffer) MakeComputeCommandEncoder() ComputeCommandEncoder {
	return ComputeCommandEncoder{CommandEncoder{C.MakeComputeCommandEncoder(b.b)}}
}

type CommandEncoder struct {
	e unsafe.Pointer
}

type ComputeCommandEncoder struct {
	CommandEncoder
}
