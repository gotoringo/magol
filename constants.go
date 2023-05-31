package magol

/*
#cgo LDFLAGS: -framework Metal -framework CoreGraphics -framework Foundation -framework MetalPerformanceShaders
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "magol.h"
*/
import "C"

// GPUFamily is a family of GPUs.
//
// See: https://developer.apple.com/documentation/metal/mtlgpufamily?language=objc.
type GPUFamily uint16

// Options for families of GPUs.
const (
	GPUFamilyApple1  GPUFamily = 1001 // Apple family 1 GPU features that correspond to the Apple A7 GPUs.
	GPUFamilyApple2  GPUFamily = 1002 // Apple family 2 GPU features that correspond to the Apple A8 GPUs.
	GPUFamilyApple3  GPUFamily = 1003 // Apple family 3 GPU features that correspond to the Apple A9 and A10 GPUs.
	GPUFamilyApple4  GPUFamily = 1004 // Apple family 4 GPU features that correspond to the Apple A11 GPUs.
	GPUFamilyApple5  GPUFamily = 1005 // Apple family 5 GPU features that correspond to the Apple A12 GPUs.
	GPUFamilyApple6  GPUFamily = 1006 // Apple family 6 GPU features that correspond to the Apple A13 GPUs.
	GPUFamilyApple7  GPUFamily = 1007 // Apple family 7 GPU features that correspond to the Apple A14 and M1 GPUs.
	GPUFamilyApple8  GPUFamily = 1008 // Apple family 8 GPU features that correspond to the Apple A15 and M2 GPUs.
	GPUFamilyMac1    GPUFamily = 2001 // Mac family 1 GPU features.
	GPUFamilyMac2    GPUFamily = 2002 // Mac family 2 GPU features.
	GPUFamilyCommon1 GPUFamily = 3001 // Common family 1 GPU features.
	GPUFamilyCommon2 GPUFamily = 3002 // Common family 2 GPU features.
	GPUFamilyCommon3 GPUFamily = 3003 // Common family 3 GPU features.
	GPUFamilyMetal3  GPUFamily = 5001 // Metal 3 features.
)
