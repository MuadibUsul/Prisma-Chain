package gemmv1

// Tiling for GEMM_INT8_V1. TILE_SIZE is fixed at 8; boundary tiles are zero
// padded in all directions and padding never contributes to
// canonical_mac_count, which is defined as M x N x K over the real
// dimensions.

// TileCounts groups the tile grid dimensions of a task.
type TileCounts struct {
	RowsA uint64 // ceil(M / 8)
	ColsA uint64 // ceil(K / 8) = R
	RowsB uint64 // ceil(K / 8) = R
	ColsB uint64 // ceil(N / 8)
	RowsC uint64 // ceil(M / 8)
	ColsC uint64 // ceil(N / 8)
}

// TileCountsFor returns the tile grid for an M x N x K task.
func TileCountsFor(m, n, k uint64) TileCounts {
	return TileCounts{
		RowsA: tileRows(m),
		ColsA: tileCols(k),
		RowsB: tileCols(k),
		ColsB: tileCols(n),
		RowsC: tileRows(m),
		ColsC: tileCols(n),
	}
}

func tileRows(dim uint64) uint64 { return (dim + TileSize - 1) / TileSize }
func tileCols(dim uint64) uint64 { return (dim + TileSize - 1) / TileSize }

// RSteps is the number of K-direction trace steps for a task: ceil(K / 8).
func RSteps(k uint64) uint64 { return tileCols(k) }

const (
	intTileCount = TileSize * TileSize // 64 int32 values per output/state tile
	intTileBytes = intTileCount * 4    // 256 canonical bytes
	int8TileSize = TileSize * TileSize // 64 int8 values per input tile
)

// ExtractATile returns the zero-padded 8x8 A tile A_tile[i][r] where i is a
// row-tile index over M and r is a column-tile index over K.
func ExtractATile(a []int8, m, k uint64, i, r uint64) [int8TileSize]int8 {
	var tile [int8TileSize]int8
	for row := uint64(0); row < TileSize; row++ {
		gi := i*TileSize + row
		if gi >= m {
			break
		}
		for col := uint64(0); col < TileSize; col++ {
			gk := r*TileSize + col
			if gk >= k {
				break
			}
			tile[row*TileSize+col] = a[gi*k+gk]
		}
	}
	return tile
}

// ExtractBTile returns the zero-padded 8x8 B tile B_tile[r][j] where r is a
// row-tile index over K and j is a column-tile index over N.
func ExtractBTile(b []int8, k, n uint64, r, j uint64) [int8TileSize]int8 {
	var tile [int8TileSize]int8
	for row := uint64(0); row < TileSize; row++ {
		gk := r*TileSize + row
		if gk >= k {
			break
		}
		for col := uint64(0); col < TileSize; col++ {
			gn := j*TileSize + col
			if gn >= n {
				break
			}
			tile[row*TileSize+col] = b[gk*n+gn]
		}
	}
	return tile
}

// Int8TileBytes returns the canonical 64 raw int8 bytes of a tile.
func Int8TileBytes(tile [int8TileSize]int8) []byte {
	out := make([]byte, int8TileSize)
	for i, v := range tile {
		out[i] = byte(v)
	}
	return out
}

// MicroStep computes Sr+1 = Sr + A_tile x B_tile for one 8x8x8 micro-step,
// exactly 512 canonical MACs with int32 accumulation.
func MicroStep(s *State, aTile, bTile [int8TileSize]int8) State {
	var next State
	for i := 0; i < TileSize; i++ {
		for j := 0; j < TileSize; j++ {
			acc := s[i*TileSize+j]
			for r := 0; r < TileSize; r++ {
				acc += int32(aTile[i*TileSize+r]) * int32(bTile[r*TileSize+j])
			}
			next[i*TileSize+j] = acc
		}
	}
	return next
}
