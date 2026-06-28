# BareSoil Unified Taxonomy

Canonical classes for BareSoil-Instruct and BareSoil-Bench.

| Unified Class | Description | Bare-positive |
|---------------|-------------|---------------|
| `bare_soil` | Exposed mineral soil, no significant vegetation | Yes |
| `sparse_vegetation` | Low cover grass, scrub, meadow with bare patches | Yes |
| `desert_sand` | Sandy desert, dunes, beaches | Yes |
| `bare_rock_paved` | Rock outcrop, paved surfaces, quarries | Yes |
| `burnt_barren` | Post-fire bare areas | Yes |
| `agricultural_fallow` | Ploughed / fallow cropland without crop cover | Yes |
| `non_bare` | Forest, urban, water, dense vegetation | No |

## Source Mappings

### AID (30-class scene)

| AID Label | Unified |
|-----------|---------|
| BareLand | bare_soil |
| Desert | desert_sand |
| Meadow | sparse_vegetation |
| Beach | desert_sand |
| Mountain | non_bare |
| Farmland | agricultural_fallow |

### LCZ42

| LCZ Label | Unified |
|-----------|---------|
| bare soil or sand | bare_soil |
| bare rock or paved | bare_rock_paved |
| low plants | sparse_vegetation |
| scrub / bush | sparse_vegetation |

### CORINE (BigEarthNet 19-class)

| CORINE Label | Unified |
|--------------|---------|
| Beaches, dunes, sands | desert_sand |
| Natural grassland and sparsely vegetated areas | sparse_vegetation |
| Arable land | agricultural_fallow |

### ESA WorldCover

| WorldCover | Unified |
|------------|---------|
| Bare / sparse vegetation | sparse_vegetation |

### Google Dynamic World

| Dynamic World | Unified |
|---------------|---------|
| Bare ground | bare_soil |

## Hard Negatives (confusers)

Include in training and benchmark: `Industrial`, `Commercial`, `Parking`, `Square`, dry rooftops, roads, shadows on vegetation.

Implementation: [`baresoil/taxonomy.py`](taxonomy.py)
