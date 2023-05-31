package magol

import (
	"fmt"

	"gorgonia.org/tensor"
)

func B(dev *Device, a, b []float32, shps ...int) (A, B, C *Matrix, TC *tensor.Dense) {
	TA := tensor.New(tensor.WithBacking(a), tensor.WithShape(shps[0], shps[1]))
	TB := tensor.New(tensor.WithBacking(b), tensor.WithShape(shps[2], shps[3]))
	TC = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(shps[0], shps[3]))

	aBuf := buf2MBuf(dev, TA)
	bBuf := buf2MBuf(dev, TB)
	cBuf := buf2MBuf(dev, TC)
	aDesc := pls(desc2MDesc(TA))
	bDesc := pls(desc2MDesc(TB))
	cDesc := pls(desc2MDesc(TC))

	A = NewMatrix(aBuf, aDesc)
	B = NewMatrix(bBuf, bDesc)
	C = NewMatrix(cBuf, cDesc)
	return
}

func Run() {

	A := tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6}), tensor.WithShape(2, 3))
	B := tensor.New(tensor.WithBacking([]float32{10, 20, 30, 40, 50, 60}), tensor.WithShape(3, 2))
	C := tensor.New(tensor.WithShape(2, 2), tensor.Of(tensor.Float32))
	//fmt.Printf("%v\n%v\n%v\n", A, B, C)

	dev := NewDevice()
	if dev == nil {
		panic("No Device Found")
	}
	aBuf := buf2MBuf(dev, A)
	bBuf := buf2MBuf(dev, B)
	cBuf := buf2MBuf(dev, C)
	aDesc := pls(desc2MDesc(A))
	bDesc := pls(desc2MDesc(B))
	cDesc := pls(desc2MDesc(C))

	matA := NewMatrix(aBuf, aDesc)
	matB := NewMatrix(bBuf, bDesc)
	matC := NewMatrix(cBuf, cDesc)

	//debug(matA)

	q := MakeCommandQueue(dev)
	cmdbuf := q.CommandBuffer()
	MPSMatMul(cmdbuf, matA, matB, matC)
	//debug(matC)
	Mbuf2Buf(C, matC.b)
	fmt.Printf("%v\n", C)
}
