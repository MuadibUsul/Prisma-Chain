# Phase F.5C baseline (A0-01)

- branch: `protocol/transformer-phase-f5c-wide-integer`
- head: `fdd92d467a76d45ed97ef1a70664c9b4a85d5f35`
- working tree clean: **False**
- toolchain: Go `windows/amd64` / Python `3.10.6`

## Test summary (all green: True)

```
?   	prismachain/cmd/prisma-gemm	[no test files]
ok  	prismachain/cmd/prisma-vm	(cached)
ok  	prismachain/compute/canonical	38.111s
ok  	prismachain/compute/gemmv1	(cached)
ok  	prismachain/compute/gemmv1/verify	(cached)
?   	prismachain/tools/f5c_canondump	[no test files]
?   	prismachain/tools/f5c_genlegacy	[no test files]
?   	prismachain/tools/f5c_replay	[no test files]
?   	prismachain/tools/gen_invsqrt_table	[no test files]
ok  	prismachain/vm	(cached)
```

```
?   	prismachain/chain/app	[no test files]
?   	prismachain/chain/cmd/prismad	[no test files]
?   	prismachain/chain/cmd/prismad/cmd	[no test files]
ok  	prismachain/chain/x/compute	(cached)
?   	prismachain/chain/x/compute/types	[no test files]
```

```
OK
```

Machine-readable copy: `docs/phase-f5c-baseline.json`.
