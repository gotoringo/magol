package magol

import (
	"math/rand"
	"testing"
	"time"

	"github.com/chewxy/math32"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func makeRandom(rows, cols int) []float32 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	retVal := make([]float32, rows*cols)
	for i := range retVal {
		x := r.ExpFloat64()
		retVal[i] = float32(x)
	}
	return retVal
}

// taken from math32, which was taken from the Go std lib
func tolerancef32(a, b, e float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}

	// note: b is correct (expected) value, a is actual value.
	// make error tolerance a fraction of b, not a.
	if b != 0 {
		e = e * b
		if e < 0 {
			e = -e
		}
	}
	return d < e
}
func closef32(a, b float32) bool      { return tolerancef32(a, b, 1e-5) } // the number gotten from the cfloat standard. Haskell's Linear package uses 1e-6 for floats
func veryclosef32(a, b float32) bool  { return tolerancef32(a, b, 1e-6) } // from wiki
func soclosef32(a, b, e float32) bool { return tolerancef32(a, b, e) }
func alikef32(a, b float32) bool {
	switch {
	case math32.IsNaN(a) && math32.IsNaN(b):
		return true
	case a == b:
		return math32.Signbit(a) == math32.Signbit(b)
	}
	return false
}

func allWithinRange(a, b []float32, fn func(a, b float32) bool) bool {
	for i := range a {
		if !fn(a[i], b[i]) {
			return false
		}
	}
	return true
}

func TestEngine_MatMul(t *testing.T) {
	d := NewDevice()
	e := NewEngine(d)
	backingA := []float32{1, 2, 3, 4, 5, 6}
	backingB := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	memA := GoSliceAsMBuf(d, backingA)
	memB := GoSliceAsMBuf(d, backingB)

	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float32), tensor.FromMemory(memA.Uintptr(), memA.MemSize()))
	b := tensor.New(tensor.WithShape(3, 3), tensor.WithEngine(e), tensor.Of(tensor.Float32), tensor.FromMemory(memB.Uintptr(), memB.MemSize()))
	c, err := a.MatMul(b)
	if err != nil {
		t.Logf("%v %v %v", a.Dtype(), b.Dtype(), c.Dtype())
		t.Fatal(err)
	}
	CC := tensor.New(tensor.WithShape(2, 3), tensor.Of(tensor.Float32))
	Mbuf2Buf(CC, c)
	assert.Equal(t, []float32{30, 36, 42, 66, 81, 96}, CC.Data())
	t.Logf("\n%v", CC)

}

func TestEngine_Add(t *testing.T) {
	d := NewDevice()
	e := NewEngine(d)
	backingA := []float32{1, 2, 3, 4, 5, 6}
	backingB := []float32{1, 2, 3, 4, 5, 6}
	memA := GoSliceAsMBuf(d, backingA)
	memB := GoSliceAsMBuf(d, backingB)

	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float32), tensor.FromMemory(memA.Uintptr(), memA.MemSize()))
	b := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float32), tensor.FromMemory(memB.Uintptr(), memB.MemSize()))

	c, err := tensor.Add(a, b)
	if err != nil {
		t.Logf("%v %v", a.Dtype(), b.Dtype())
		t.Fatal(err)
	}
	CC := tensor.New(tensor.WithShape(2, 3), tensor.Of(tensor.Float32))
	Mbuf2Buf(CC, c)
	assert.Equal(t, []float32{2, 4, 6, 8, 10, 12}, CC.Data())
	t.Logf("\n%v", CC)
}

func TestEngine_SoftMax(t *testing.T) {
	r := 2
	c := 3
	d := NewDevice()
	e := NewEngine(d)

	backingA := makeRandom(r, c)
	memA := GoSliceAsMBuf(d, backingA)
	a := tensor.New(tensor.WithShape(r, c), tensor.WithEngine(e), tensor.Of(tensor.Float32), tensor.FromMemory(memA.Uintptr(), memA.MemSize()))
	b, err := tensor.SoftMax(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	BB := tensor.New(tensor.WithShape(r, c), tensor.Of(tensor.Float32))
	Mbuf2Buf(BB, b)

	backingX := make([]float32, r*c)
	copy(backingX, backingA)
	x := tensor.New(tensor.WithShape(r, c), tensor.WithBacking(backingX))
	y, err := tensor.SoftMax(x, 1)
	if err != nil {
		t.Fatal(err)
	}

	backingS := make([]float32, r*c)
	copy(backingS, backingA)
	s := mat.NewDense(r, c, backingS)
	ms := make([]mat.Matrix, 0, r)
	rs := make([]mat.Matrix, 0, r)
	for i := 0; i < 2; i++ {
		ms = append(ms, s.ExtractRow(i))
	}
	for _, r := range ms {
		rs = append(rs, r.Softmax())
	}
	var recombined []float32
	for _, r := range rs {
		recombined = append(recombined, r.Data().F32()...)
	}

	assert.True(t, allWithinRange(BB.Data().([]float32), y.Data().([]float32), veryclosef32), "metal ≠ tensor")
	assert.True(t, allWithinRange(recombined, y.Data().([]float32), veryclosef32), "spago ≠ tensor")
	assert.True(t, allWithinRange(BB.Data().([]float32), recombined, veryclosef32), "spago ≠ metal")

}
