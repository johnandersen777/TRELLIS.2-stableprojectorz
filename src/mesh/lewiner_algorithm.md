# Lewiner Marching Cubes Algorithm Specification

Based on: scikit-image `_marching_cubes_lewiner_cy.pyx` (1445 lines Cython) + `_marching_cubes_lewiner_luts.py` (672 lines lookup tables)

## 1. Algorithm Overview

The Lewiner marching cubes algorithm is an improved version of Chernyaev's Marching Cubes 33. It guarantees topologically correct results by resolving ambiguous face configurations through sub-case resolution.

**Reference:** Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira, Geovan Tavares. "Efficient implementation of Marching Cubes' cases with topological guarantees." Journal of Graphics Tools 8(2): pp. 1-15 (December 2003).

---

## 2. Entry Point: `marching_cubes()`

### Signature (Cython)
```python
def marching_cubes(
    float32[:,:,:] im,       # 3D volume data (Nz, Ny, Nx)
    float64 isovalue,         # Contour value
    LutProvider luts,         # Preloaded lookup tables
    int st=1,                 # Step size (voxels)
    int classic=0,            # 0 = Lewiner, 1 = classic (Lorensen)
    ndarray mask=None         # Optional boolean mask
) -> (vertices, faces, normals, values)
```

### Python wrapper (`_marching_cubes_lewiner.py`)
Before calling Cython:
- Volume is converted to `float32` contiguous array: shape (Nz, Ny, Nx)
- Isovalue is computed as `0.5 * (min + max)` if not specified
- `spacing` is applied after algorithm (multiply vertex positions)
- `gradient_direction` flips face winding if 'descent'
- `allow_degenerate=False` calls `remove_degenerate_faces` post-process
- Returns: vertices (V,3), faces (F,3), normals (V,3), values (V,)

### Post-processing steps
1. Vertices are flipped: `np.fliplr(vertices)` (z-y-x -> x-y-z)
2. Faces reshaped to (-1, 3)
3. If `gradient_direction == 'descent'`: faces are flipped: `np.fliplr(faces)`
4. If `spacing != (1,1,1)`: vertices are scaled
5. If `!allow_degenerate`: degenerate faces are removed

---

## 3. Cell Classification: Cube Index Computation

### Cube Index (8-bit mask)

Each cube has 8 corners. For each corner with value > isovalue, the corresponding bit is set:

```
// Corner numbering (after subtracting isovalue from each)
index = 0
if v0 > 0:  index += 1      // bit 0
if v1 > 0:  index += 2      // bit 1
if v2 > 0:  index += 4      // bit 2
if v3 > 0:  index += 8      // bit 3
if v4 > 0:  index += 16     // bit 4
if v5 > 0:  index += 32     // bit 5
if v6 > 0:  index += 64     // bit 6
if v7 > 0:  index += 128    // bit 7
```

This produces a value 0-255 representing which corners are inside (>isovalue) vs outside.

### Phase angle (side note)
The Lewiner paper uses a "phase angle" concept but the Cython implementation does **not** implement it. The code path is always "non-classic" (classic=0).

### Corner numbering and mapping

```
       7 ________ 6          Corners (x,y,z):
       /|       /|            v0 = (0,0,0), v1 = (1,0,0)
      /  |     /  |           v2 = (1,1,0), v3 = (0,1,0)
   4 /_______ /    |          v4 = (0,0,1), v5 = (1,0,1)
    |     |  |5    |          v6 = (1,1,1), v7 = (0,1,1)
    |    3|__|_____|2
    |    /   |    /
    |  /     |  /
    |/_______|/
   0          1
```

**IMPORTANT:** The `vv[]` array (used for interpolation) has a different ordering:
```
vv[0] = v0  (0,0,0)     vv[4] = v4  (0,0,1)
vv[1] = v1  (1,0,0)     vv[5] = v5  (1,0,1)
vv[2] = v3  (0,1,0)     vv[6] = v7  (0,1,1)   // note: v3/v2 and v7/v6 swapped
vv[3] = v2  (1,1,0)     vv[7] = v6  (1,1,1)
```

The mapping is: `i = dz*4 + dy*2 + dx` (a bit-interleaved pattern).

---

## 4. The CASES Lookup Table

### Structure
`CASES` is a `(256, 2)` table indexed by cube index (0-255).

```
caseIndex = CASES[cell.index][0]   // The case number 1-14 (0 = no surface)
config    = CASES[cell.index][1]   // The sub-configuration within that case
```

- If `caseIndex == 0`: no surface passes through this cell (skip it)
- If `caseIndex > 0`: pass to `the_big_switch()` along with `config`

### Cases 1-14

The 14 base cases from Chernyaev's Marching Cubes 33:
| Case | Name | Triangles produced | Notes |
|------|------|-------------------|-------|
| 0 | Empty | 0 | No surface |
| 1 | One triangle | 1 | Simplest case |
| 2 | Quadrilateral split | 2 | 2 triangles |
| 3a/3b | Triangle+quad ambiguous | 2 or 4 | Face test needed |
| 4a/4b | 2 triangles or complex | 2 or 6 | Internal test needed |
| 5 | 3 triangles | 3 | Resolved |
| 6a/6b/6c | 3, 9, or 5 triangles | 3,9,5 | Face + internal test |
| 7a-7h | 3, 5, 9 triangles | varies | 3 face tests + subconfig |
| 8 | 2 triangles | 2 | Opposite corners |
| 9 | 4 triangles | 4 | |
| 10a-10d | 4 or 8 triangles | 4,8 | Face + internal test |
| 11 | 4 triangles | 4 | |
| 12a-12d | 4 or 8 triangles | 4,8 | Face + internal test |
| 13a-13v | 4-12 triangles | varies | 6 face tests + subconfig |
| 14 | 4 triangles | 4 | |

### CASESCLASSIC Table (for `classic=1` path)
`CASESCLASSIC` is a `(256, 16)` table. Each entry is a sequence of vertex indices (edge numbers, 0-11, or -1 terminator):
```
nt = 0
while CASESCLASSIC[cell.index][3*nt] != -1:
    nt += 1
// Then emit triangles: read triples of edge indices
```

This is the traditional (Lorensen) marching cubes output -- no sub-case resolution, no ambiguity handling.

---

## 5. The Big Switch: `the_big_switch()`

```c
void the_big_switch(LutProvider luts, Cell cell, int case, int config):
    subconfig = 0

    if case == 1:
        add_triangles(TILING1, config, 1)         // fixed: 1 triangle
    elif case == 2:
        add_triangles(TILING2, config, 2)         // fixed: 2 triangles
    elif case == 3:
        if test_face(cell, TEST3[config]):
            add_triangles(TILING3_2, config, 4)   // 4 triangles
        else:
            add_triangles(TILING3_1, config, 2)   // 2 triangles
    elif case == 4:
        if test_internal(cell, TEST4[config]):
            add_triangles(TILING4_1, config, 2)   // 2 triangles
        else:
            add_triangles(TILING4_2, config, 6)   // 6 triangles
    elif case == 5:
        add_triangles(TILING5, config, 3)         // fixed: 3 triangles
    elif case == 6:
        if test_face(cell, TEST6[config][0]):
            add_triangles(TILING6_2, config, 5)   // 5 triangles
        else:
            if test_internal(cell, TEST6[config][1]):
                add_triangles(TILING6_1_1, config, 3)      // 3 triangles
            else:
                add_triangles(TILING6_1_2, config, 9)      // 9 triangles (center vertex)
    // ... cases 7-14 follow same pattern
```

### Face Test: `test_face(Cell, face)`

Determines which of two iso-surface configurations a face has (ambiguous face resolution).

```python
def test_face(cell, face):
    absFace = abs(face)
    # Select four corners of the face
    if absFace == 1:  A,B,C,D = v0, v4, v5, v1  # front face (y=0)
    if absFace == 2:  A,B,C,D = v1, v5, v6, v2  # right face (x=1)
    if absFace == 3:  A,B,C,D = v2, v6, v7, v3  # back face (y=1)
    if absFace == 4:  A,B,C,D = v3, v7, v4, v0  # left face (x=0)
    if absFace == 5:  A,B,C,D = v0, v3, v2, v1  # bottom face (z=0)
    if absFace == 6:  A,B,C,D = v4, v7, v6, v5  # top face (z=1)

    AC_BD = A*C - B*D
    if abs(AC_BD) < EPSILON:
        return face >= 0          # ambiguous: return face sign
    else:
        return face * A * AC_BD >= 0  # resolve ambiguity
```

The sign of `face` controls which side of the ambiguous face to pick.

### Internal Test: `test_internal(Cell, case, config, subconfig, s)`

Resolves ambiguous internal configurations for cases 4, 6, 7, 10, 12, 13.

For cases 4 and 10:
- Computes a parameter `t` along the body diagonal
- Interpolates 4 edge values (At, Bt, Ct, Dt)
- Counts non-negative values -> `test` bitmask (0-15)
- Uses the bitmask to determine result

For cases 6, 7, 12, 13:
- References a specific **reference edge** from TEST6/TEST7/TEST12/TILING13_5_1 tables
- Computes parameter `t` along that edge
- Interpolates values around the cell
- Same `test` bitmask -> result logic

The final logic maps `test` (0-15) to boolean:
```
test=0-4: return s>0
test=5:   if At*Ct - Bt*Dt < EPSILON: return s>0
test=6:   return s>0
test=7:   return s<0
test=8-9: return s>0
test=10:  if At*Ct - Bt*Dt >= EPSILON: return s>0
test=11:  return s<0
test=12:  return s>0
test=13-15: return s<0
```

### Sub-configuration resolution

For cases 7 and 13, multiple face tests produce a `subconfig` bitmask:

**Case 7:** 3 face tests => subconfig bits 0-2 (values 0-7)
- TILING7_1 through TILING7_4_2 indexed by subconfig

**Case 13:** 6 face tests => subconfig bits 0-5 (values 0-63)
- First, subconfig is built from 6 test_face calls
- Then `subconfig = SUBCONFIG13[subconfig]` maps 0-63 to a compressed value (0-45)
- The compressed subconfig selects among TILING13_1 through TILING13_5_2

**Case 13 subconfig mapping:**
- 0: TILING13_1 (4 triangles, 12 indices)
- 1-6: TILING13_2 (6 triangles, 18 indices) -- 6 face varieties
- 7-18: TILING13_3 (10 triangles, 30 indices) -- 12 center-vertex varieties
- 19-22: TILING13_4 (12 triangles, 36 indices) -- 4 center-vertex varieties
- 23-26: TILING13_5_1 / TILING13_5_2 (6 or 10 triangles) -- internal test
- 27-38: TILING13_3_ (10 triangles, 30 indices) -- 12 mirrored varieties
- 39-44: TILING13_2_ (6 triangles, 18 indices) -- 6 mirrored varieties
- 45: TILING13_1_ (4 triangles, 12 indices)

---

## 6. Tiling Tables: How Sub-Cases Produce Triangles

### Table naming convention

TILING tables encode edge indices (0-12, where 12 = center vertex). They are indexed by:
- `config` (the per-case configuration index)
- `subconfig` (for multi-variant cases)

### Table shapes

| Table | Shape | Elements | Description |
|-------|-------|----------|-------------|
| TILING1 | (16, 3) | 1 triangle | Cases with 1 config, 1 triangle |
| TILING2 | (24, 6) | 2 triangles | Cases with 1 config, 2 triangles |
| TILING3_1 | (24, 6) | 2 triangles | Face-test=false, 2 triangles |
| TILING3_2 | (24, 12) | 4 triangles | Face-test=true, 4 triangles |
| TILING4_1 | (8, 6) | 2 triangles | Internal-test=true |
| TILING4_2 | (8, 18) | 6 triangles | Internal-test=false |
| TILING5 | (48, 9) | 3 triangles | Fixed geometry |
| TILING6_1_1 | (48, 9) | 3 triangles | Face=false, Internal=true |
| TILING6_1_2 | (48, 27) | 9 triangles | Face=false, Internal=false (center) |
| TILING6_2 | (48, 15) | 5 triangles | Face=true |
| TILING7_1 | (16, 9) | 3 triangles | Subconfig=0 |
| TILING7_2 | (16, 3, 15) | 5 triangles | Subconfig=1,2,4 |
| TILING7_3 | (16, 3, 27) | 9 triangles | Subconfig=3,5,6 (center) |
| TILING7_4_1 | (16, 15) | 5 triangles | Subconfig=7, Internal=true |
| TILING7_4_2 | (16, 27) | 9 triangles | Subconfig=7, Internal=false |
| TILING8 | (6, 6) | 2 triangles | Fixed |
| TILING9 | (8, 12) | 4 triangles | Fixed |
| TILING10_1_1 | (6, 12) | 4 triangles | |
| TILING10_1_1_ | (6, 12) | 4 triangles | Mirror of 10_1_1 |
| TILING10_1_2 | (6, 24) | 8 triangles | |
| TILING10_2 | (6, 24) | 8 triangles | |
| TILING10_2_ | (6, 24) | 8 triangles | |
| TILING11 | (12, 12) | 4 triangles | Fixed |
| TILING12_1_1 | (24, 12) | 4 triangles | |
| TILING12_1_1_ | (24, 12) | 4 triangles | |
| TILING12_1_2 | (24, 24) | 8 triangles | |
| TILING12_2 | (24, 24) | 8 triangles | |
| TILING12_2_ | (24, 24) | 8 triangles | |
| TILING13_1 | (2, 12) | 4 triangles | |
| TILING13_1_ | (2, 12) | 4 triangles | |
| TILING13_2 | (2, 6, 18) | 6 triangles | |
| TILING13_2_ | (2, 6, 18) | 6 triangles | |
| TILING13_3 | (2, 12, 30) | 10 triangles | |
| TILING13_3_ | (2, 12, 30) | 10 triangles | |
| TILING13_4 | (2, 4, 36) | 12 triangles | |
| TILING13_5_1 | (2, 4, 18) | 6 triangles | |
| TILING13_5_2 | (2, 4, 30) | 10 triangles | |
| TILING14 | (12, 12) | 4 triangles | Fixed |

### How tiling values encode vertices

Each entry in a tiling table is a **signed byte** that represents an edge index (0-11) or the center vertex (12).

- Values >= 0: edge/vertex index
- Value -1: terminator (end of triangle list for CASESCLASSIC)
- The edge indices correspond to the 12 edges + center (12) of the cube

### How `add_triangles` works

```python
def add_triangles(Lut lut, int lutIndex, int nt):
    # lutIndex = config (or other selector)
    # nt = known number of triangles to emit
    cell.prepare_for_adding_triangles()
    for i in range(nt):
        for j in range(3):  # 3 vertices per triangle
            vi = lut.get2(lutIndex, i*3 + j)  # edge index
            cell._add_face_from_edge_index(vi)
```

For `add_triangles2`:
```python
def add_triangles2(Lut lut, lutIndex, lutIndex2, nt):
    cell.prepare_for_adding_triangles()
    for i in range(nt):
        for j in range(3):
            vi = lut.get3(lutIndex, lutIndex2, i*3 + j)  # 3D lookup
            cell._add_face_from_edge_index(vi)
```

---

## 7. Vertex Computation and Deduplication

### `_add_face_from_edge_index(vi)`

This is the core method that creates/interpolates vertices and deduplicates them.

#### For center vertex (vi == 12):
1. If not pre-calculated, call `calculate_center_vertex()` (center-of-mass interpolation)
2. Look up `indexInVertexArray` from the face layer
3. If vertex exists (`>= 0`): only add face reference and gradient
4. If vertex doesn't exist: create new vertex at interpolated position, store in face layer, add face + gradient

#### For edge vertices (vi < 12):
1. Get relative edge indices from `EDGESRELX`, `EDGESRELY`, `EDGESRELZ` tables:
   - Each table provides two relative positions (dx, dy, dz) per edge
   - These map to two corner indices: `index1 = dz1*4 + dy1*2 + dx1`
2. Compute interpolation weights:
   ```
   tmpf1 = 1.0 / (EPSILON + abs(vv[index1]))
   tmpf2 = 1.0 / (EPSILON + abs(vv[index2]))
   ```
3. Look up `indexInVertexArray` from the face layer
4. If vertex exists: add face reference and accumulate gradient contributions
5. If not: interpolate position using center-of-mass method:
   ```
   fx = dx1*tmpf1 + dx2*tmpf2   // positions weighted by inverse absolute value
   fy = dy1*tmpf1 + dy2*tmpf2
   fz = dz1*tmpf1 + dz2*tmpf2
   ff = tmpf1 + tmpf2
   // Vertex at: (x + step*fx/ff, y + step*fy/ff, z + step*fz/ff)
   ```
6. Store resulting vertex index in face layer for dedup

### Vertex Deduplication via Face Layers

The key insight: vertices on edges are shared between adjacent cells. The algorithm uses **face layers** to cache edge vertices per cell.

#### `get_index_in_facelayer(vi)`

Maps an edge index to a unique slot `(cell_index, j)` in the face layer array:

- **Edges 0-3** (bottom horizontal): stored in `faceLayer1`, 4 per cell
  - Edge 0: `(x, y, z)` -> slot 0
  - Edge 1: `(x+step, y, z)` -> slot 1
  - Edge 2: `(x, y+step, z)` -> slot 2  
  - Edge 3: `(x, y, z)` -> slot 1
- **Edges 4-7** (top horizontal): stored in `faceLayer2` (same pattern)
- **Edges 8-11** (vertical): stored in `faceLayer1`, 4 per cell
- **Edge 12** (center): stored in `faceLayer1`, 4 per cell

The face layer arrays have size `nx * ny * 4` (4 slots per cell). Each slot holds either -1 (no vertex yet) or a vertex array index.

When a new z-layer starts, `new_z_value()` swaps `faceLayer1` and `faceLayer2`, and clears the new `faceLayer2`.

This scheme ensures:
- Horizontal edges in z are shared across z-layers via the swap
- Vertical edges are shared within the same z-slice
- Center vertices are unique per cell

---

## 8. Cell Traversal

### Loop structure

```python
z = -step
while z < Nz - 2*step:
    z += step
    cell.new_z_value()  # swap face layers, clear new layer
    
    y = -step
    while y < Ny - 2*step:
        y += step
        
        x = -step
        while x < Nx - 2*step:
            x += step
            
            if no_mask or mask[z+step, y+step, x+step]:
                # Read 8 corner values
                cell.set_cube(isovalue, x, y, z, step,
                    im[z,   y,   x],    im[z,   y,   x+step],
                    im[z,   y+step, x+step], im[z,   y+step, x],
                    im[z+step, y,   x],    im[z+step, y,   x+step],
                    im[z+step, y+step, x+step], im[z+step, y+step, x])
                
                if classic:
                    # CASESCLASSIC path
                else:
                    # Lewiner path
                    case = CASES[cell.index][0]
                    if case > 0:
                        config = CASES[cell.index][1]
                        the_big_switch(luts, cell, case, config)
```

### Boundary condition

Valid cells span from `x, y, z = -step` to `Nx-2*step, Ny-2*step, Nz-2*step`. The cell at position `(x, y, z)` reads corners at `(x, y, z)` through `(x+step, y+step, z+step)`, so the outermost voxel layer (last `step`-wide layer) acts as a guard band.

### Step size

`step` controls the stride. Larger steps = fewer cells = coarser mesh but faster. The algorithm remains topologically correct because it skips entire cells rather than subsampling the volume.

### Mask support

If a mask array is provided, cells are only processed when `mask[z+step, y+step, x+step]` is True (center voxel of cell). This allows "cut-out" regions where no surface is generated.

---

## 9. Center Vertex Calculation

### `calculate_center_vertex()`

The center vertex (edge 12) is used when case resolution requires an interior point (subcases with the `// v12 needed` comment in the_big_switch).

The position is interpolated using inverse-value-weighted center-of-mass:

```
v_n = 1 / (EPSILON + abs(corner_n))  // weight for each corner

fx = 0*v0 + 1*v1 + 1*v2 + 0*v3 + 0*v4 + 1*v5 + 1*v6 + 0*v7
fy = 0*v0 + 0*v1 + 1*v2 + 1*v3 + 0*v4 + 0*v5 + 1*v6 + 1*v7
fz = 0*v0 + 0*v1 + 0*v2 + 0*v3 + 1*v4 + 1*v5 + 1*v6 + 1*v7
ff = sum of all v_n

center_x = cell.x + step * fx / ff
center_y = cell.y + step * fy / ff
center_z = cell.z + step * fz / ff
```

The gradient at center is a weighted sum of the 8 corner gradients using the same weights.

---

## 10. Gradient/Normal Computation

### `prepare_for_adding_triangles()`

Called once per cell before emitting triangles. Computes:

1. **`vv[]` array**: copies the 8 corner values into indexable form (with the v2/v3 and v6/v7 swap)
2. **`vmax`**: range of values (max - min), used for the `values` output
3. **Gradients at corners**: central differences approximated between adjacent corners:

```
gradient[0] = (v0-v1, v0-v3, v0-v4)  // d/dx, d/dy, d/dz for corner 0
gradient[1] = (v0-v1, v1-v2, v1-v5)
// ... etc for all 8 corners
```

### Gradient accumulation

Each time a vertex is referenced from a cell (either newly created or reused), its gradient is accumulated:

```python
add_gradient_from_index(vertexIndex, cornerIndex, strength):
    normals[vertexIndex] += gradient[cornerIndex] * strength
```

The `strength` is `1 / (EPSILON + abs(vv[cornerIndex]))` -- corners with values closer to zero contribute more to the normal.

### Normal finalization

After all cells are processed, normals are normalized:
```
length = sqrt(nx^2 + ny^2 + nz^2)
if length > 0: length = 1 / length
normal = (nx * length, ny * length, nz * length)
```

---

## 11. Memory Management (Cell Class)

### Dynamic arrays
- `_vertices`: float32, 3 components per vertex, starts at 8, doubles when full
- `_normals`: float32, 3 components per vertex, same growth
- `_values`: float32, 1 component per vertex, same growth
- `_faces`: int32, 1 component per face, starts at 8, doubles when full

### Face layers
- `faceLayer1`, `faceLayer2`: arrays of size `nx * ny * 4` initialized to -1
- On new z-layer: swap and clear the second layer

### References
- No manual memory management in TypeScript (use arrays/typed arrays)

---

## 12. Pseudo-code Summary

```
function marching_cubes(volume, isovalue, step=1, classic=false, mask=null):
    (Nz, Ny, Nx) = volume.shape
    cell = new Cell(Nx, Ny, Nz)
    load_lookup_tables()

    for z from -step to Nz - 2*step, step:
        cell.new_z_layer()
        for y from -step to Ny - 2*step, step:
            for x from -step to Nx - 2*step, step:
                if mask and !mask[z+step][y+step][x+step]: continue

                // Read 8 corner values
                corners = read_cube_corners(volume, x, y, z, step)
                cell.set_cube(isovalue, x, y, z, step, corners)

                if cell.index == 0: continue  // no surface

                if classic:
                    case_entries = CASESCLASSIC[cell.index]
                    nt = count_triangles(case_entries)
                    if nt > 0:
                        cell.add_triangles(case_entries, nt)
                else:
                    case = CASES[cell.index][0]
                    config = CASES[cell.index][1]
                    the_big_switch(cell, case, config)

    return (cell.vertices, cell.faces, cell.normals, cell.values)


function the_big_switch(cell, case, config):
    subconfig = 0

    switch case:
        1:  add_triangles(TILING1[config], 1)
        2:  add_triangles(TILING2[config], 2)
        3:  if test_face(TEST3[config]):
                add_triangles(TILING3_2[config], 4)
            else:
                add_triangles(TILING3_1[config], 2)
        4:  if test_internal(TEST4[config]):
                add_triangles(TILING4_1[config], 2)
            else:
                add_triangles(TILING4_2[config], 6)
        5:  add_triangles(TILING5[config], 3)
        6:  if test_face(TEST6[config][0]):
                add_triangles(TILING6_2[config], 5)
            else:
                if test_internal(TEST6[config][1]):
                    add_triangles(TILING6_1_1[config], 3)
                else:
                    add_triangles(TILING6_1_2[config], 9)  // uses center
        7:  subconfig = test_face(TEST7[config][0]) * 1
                + test_face(TEST7[config][1]) * 2
                + test_face(TEST7[config][2]) * 4
            // 8 subcases based on subconfig 0-7
            // Uses TILING7_1, TILING7_2, TILING7_3, TILING7_4_1, TILING7_4_2
        8:  add_triangles(TILING8[config], 2)
        9:  add_triangles(TILING9[config], 4)
        10: // Two face tests + internal test
            // TILING10_1_1, TILING10_1_1_, TILING10_1_2, TILING10_2, TILING10_2_
        11: add_triangles(TILING11[config], 4)
        12: // Two face tests + internal test
            // TILING12_1_1, TILING12_1_1_, etc.
        13: subconfig = test_face(TEST13[config][0]) * 1
                + test_face(TEST13[config][1]) * 2
                + test_face(TEST13[config][2]) * 4
                + test_face(TEST13[config][3]) * 8
                + test_face(TEST13[config][4]) * 16
                + test_face(TEST13[config][5]) * 32
            subconfig = SUBCONFIG13[subconfig]
            // 46 subcases from TILING13_1 through TILING13_5_2
        14: add_triangles(TILING14[config], 4)


function add_triangles(lut, config, nt):
    cell.prepare_for_adding_triangles()
    for i in 0..nt:
        for j in 0..2:
            edge_index = lut[config][i*3+j]  // edge 0-11 or 12(center)
            cell._add_face_from_edge_index(edge_index)


function _add_face_from_edge_index(edge_index):
    slot = cell.get_face_layer_slot(edge_index)
    existing_vertex = cell.faceLayer[slot]

    if edge_index == 12:    // center vertex
        if not v12_calculated:
            calculate_center_vertex()
        if existing_vertex >= 0:
            cell.add_face(existing_vertex)
            cell.add_gradient(existing_vertex, v12_grad)
        else:
            idx = cell.add_vertex(v12_x, v12_y, v12_z)
            cell.faceLayer[slot] = idx
            cell.add_face(idx)
            cell.add_gradient(idx, v12_grad)

    else:   // edge vertex 0-11
        (dx1,dx2) = EDGESRELX[edge_index]
        (dy1,dy2) = EDGESRELY[edge_index]
        (dz1,dz2) = EDGESRELZ[edge_index]

        corner1 = dz1*4 + dy1*2 + dx1
        corner2 = dz2*4 + dy2*2 + dx2

        w1 = 1.0 / (EPSILON + abs(vv[corner1]))
        w2 = 1.0 / (EPSILON + abs(vv[corner2]))

        if existing_vertex >= 0:
            cell.add_face(existing_vertex)
            cell.add_gradient(existing_vertex, corner1, w1)
            cell.add_gradient(existing_vertex, corner2, w2)
        else:
            // Interpolate edge vertex
            fx = dx1*w1 + dx2*w2
            fy = dy1*w1 + dy2*w2
            fz = dz1*w1 + dz2*w2
            fw = w1 + w2

            pos_x = cell.x + step * fx / fw
            pos_y = cell.y + step * fy / fw
            pos_z = cell.z + step * fz / fw

            idx = cell.add_vertex(pos_x, pos_y, pos_z)
            cell.faceLayer[slot] = idx
            cell.add_face(idx)
            cell.add_gradient(idx, corner1, w1)
            cell.add_gradient(idx, corner2, w2)
```

---

## 13. TypeScript Mapping Plan

### Types

| Cython type | TypeScript equivalent |
|-------------|----------------------|
| `float64` | `number` (f64) |
| `float32` | `number` (stored in Float32Array) |
| `int` | `number` (i32) |
| `signed char` | `number` (i8) |
| `float64_t *` | `Float64Array` |
| `float32_t *` | `Float32Array` |
| `int *` | `Int32Array` |
| `cnp.ndarray` | `Float32Array` (for volume) |
| `char[][][]` (LUT) | `Int8Array` with linear indexing |

### Key data structures

```typescript
class Cell {
    x: number; y: number; z: number; step: number;
    v0-v7: number;  // corner values (isovalue-subtracted)
    vv: Float64Array;  // [8] re-indexed corners
    vg: Float64Array;  // [8*3] corner gradients
    vmax: number;
    v12: { x, y, z, xg, yg, zg, calculated: boolean };
    index: number;  // 8-bit mask
    nx: number; ny: number; nz: number;
    faceLayer1: Int32Array;  // [nx*ny*4]
    faceLayer2: Int32Array;
    faceLayer: Int32Array;  // reference to current
    vertices: Float32Array;  // dynamic growth
    normals: Float32Array;
    values: Float32Array;
    faces: Int32Array;
    vertexCount: number; faceCount: number;
}

class Lut {
    data: Int8Array;  // flattened table
    L0: number; L1: number; L2: number;  // dimensions
    
    get1(i0): number;
    get2(i0, i1): number;
    get3(i0, i1, i2): number;
}

class LutProvider {
    EDGESRELX, EDGESRELY, EDGESRELZ: Lut;
    CASESCLASSIC, CASES: Lut;
    TILING1..TILING14: Lut;
    TEST3..TEST13: Lut;
    SUBCONFIG13: Lut;
}
```

### LUT Decoding

The LUT tables in `_marching_cubes_lewiner_luts.py` are base64-encoded. In TypeScript, pre-decode them once:

1. Decode base64 to Uint8Array
2. Interpret as signed Int8Array
3. Reshape to the declared dimensions

Example decoding for CASES (256, 2):
```typescript
function decodeLUT(shape: number[], encoded: string): Int8Array {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const signed = new Int8Array(bytes.buffer);
    return signed;  // flat array, index manually using shape
}
```

### Performance considerations

1. **Use typed arrays throughout** (Float32Array, Int32Array, Int8Array)
2. **Avoid object allocations in inner loops** -- pre-allocate all scratch buffers
3. **The main loop** is the triple-nested x/y/z traversal -- minimize work per cell
4. **Face layer arrays** (`nx * ny * 4`) can be large for big volumes
5. **Dynamic growth** of vertex/face arrays should use exponential strategy (x2)
6. **LUT access** should be `Int8Array[index]` with precomputed strides
7. **ES module** with all LUTs encoded as base64 strings

---

## 14. `remove_degenerate_faces` Post-Process

Called when `allow_degenerate=false`:

```typescript
function removeDegenerateFaces(
    vertices: Float32Array,  // [V][3]
    faces: Int32Array,       // [F][3]
    ...arrays: Float32Array[]  // normals, values, etc.
): [Float32Array, Int32Array, ...Float32Array[]]
```

Algorithm:
1. Initialize `vertexMap = [0,1,2,...,V-1]` and `faces_ok = [1]*F`
2. For each face, check if any two vertices are identical:
   - If vertex i1 == i2: merge vertexMap entries to min index, mark face bad
3. After all faces: `vertices_ok = vertexMap[i] == i` (unique vertices)
4. Build `vertexMap2 = prefixSum(vertices_ok) - 1` (new indices)
5. Remap faces: `newFaces = vertexMap2[vertexMap[faces[faces_ok]]]`
6. Select kept vertices: `newVertices = vertices[vertices_ok]`
7. Same for other arrays

---

## 15. Edge Index Reference

```
Edge 0:  v0(0,0,0) - v1(1,0,0)   // bottom, front
Edge 1:  v1(1,0,0) - v2(1,1,0)   // bottom, right
Edge 2:  v2(1,1,0) - v3(0,1,0)   // bottom, back
Edge 3:  v3(0,1,0) - v0(0,0,0)   // bottom, left
Edge 4:  v4(0,0,1) - v5(1,0,1)   // top, front
Edge 5:  v5(1,0,1) - v6(1,1,1)   // top, right
Edge 6:  v6(1,1,1) - v7(0,1,1)   // top, back
Edge 7:  v7(0,1,1) - v4(0,0,1)   // top, left
Edge 8:  v0(0,0,0) - v4(0,0,1)   // vertical, front-left
Edge 9:  v1(1,0,0) - v5(1,0,1)   // vertical, front-right
Edge 10: v2(1,1,0) - v6(1,1,1)   // vertical, back-right
Edge 11: v3(0,1,0) - v7(0,1,1)   // vertical, back-left
Edge 12: Center of cell            // interior
```

### EDGETORELATIVEPOS arrays

```
// For each edge (0-11), two (dx, dy, dz) endpoints
EDGESRELX: [0,1], [1,1], [1,0], [0,0],   // edges 0-3: x coords
            [0,1], [1,1], [1,0], [0,0],   // edges 4-7
            [0,0], [1,1], [1,1], [0,0]    // edges 8-11

EDGESRELY: [0,0], [0,1], [1,1], [1,0],   // edges 0-3: y coords
            [0,0], [0,1], [1,1], [1,0],   // edges 4-7
            [0,0], [0,0], [1,1], [1,1]    // edges 8-11

EDGESRELZ: [0,0], [0,0], [0,0], [0,0],   // edges 0-3: z coords
            [1,1], [1,1], [1,1], [1,1],   // edges 4-7
            [0,1], [0,1], [0,1], [0,1]    // edges 8-11
```

Note: These represent relative coordinates within the cell's unit cube (0 or 1 in each dimension). The actual vertex position is:
```
corner_index = dz*4 + dy*2 + dx
```
This maps (dx,dy,dz) to index in `vv[]` array.

---

## 16. Complete File List for Port

| File | Purpose |
|------|---------|
| `marching_cubes_lewiner_cy.pyx` | Main algorithm (1445 lines) |
| `marching_cubes_lewiner_luts.py` | All lookup tables encoded (672 lines) |
| `marching_cubes_lewiner.py` | Python wrapper + LUT initialization (353 lines) |

**Total: ~2470 lines of Cython/Python to port.**

The LUT tables in `_marching_cubes_lewiner_luts.py` are static data that must be embedded (probably as base64 strings decoded once at module load time). The ~2470 lines are dominated by the LUT definitions -- the actual algorithm logic is ~400 lines.
