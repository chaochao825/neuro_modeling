# Exp28 frozen-ceiling independent source result

This directory preserves the fail-closed result for the originally frozen
Exp28 source protocol (`max_scale=128`, seeds 30--59). It is source-only and
performs no selector inference.

- Run commit: `e650dd17554d62124f714e52a1ab7d171fa3f2b2`
- Run tree: `30d7939386c4ec218c5b74ef92e35f645b571224`
- Registered rows: 13,200
- Complete / failed / invalid: 13,199 / 1 / 0
- Source panel valid: `false`
- Frozen-ceiling portability conclusion: **oppose**
- Confirmatory selector conclusion: **inconclusive** (the registered source
  gate did not pass, so the selector was not run)

The sole failure is seed 52, generator `d65464a1a0917550a226`, actuator
`routing`, alpha 1.0, transition rank 1, input rank 2, delay 12, and noise
standard deviation 0.3. The recorded error is `ActuatorFitError: functional-
budget scale is non-finite or exceeds max_scale`. The failure is in the
discovery split; it is retained and is not imputed or reclassified.

`exp28_source_v3_e650dd1_invalid_package.tar.gz` is a deterministic archive of
the full package, including the 101,489,496-byte canonical raw JSONL. Its
SHA-256 is
`107039999aaa271d27aafd97746d74e76eb341be15247d4a84994a884a7d7c10`.
The uncompressed raw JSONL SHA-256 recorded by the package is
`55e825667f003d5f80b489eb13cdecb57dcb4a40aab10b0c68250e69b37632fe`.

The receipt and conclusion file hashes are, respectively,
`d01352232ac22d35153ebd74e4916bbb8e9b598ddcb622b269fa73d6733be96d`
and `07ea53eb824d77f1d53699f73d1cc158b038c735fb0e67c3cf68833716660ff0`.
Any later ceiling-256 run is a separately versioned post-hoc sensitivity and
does not overwrite or restore the confirmatory status of this result.
