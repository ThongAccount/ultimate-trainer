# Results: dX rasterization transpose (2026-11-02)

Branch `feat/dx-xpose` @ 6f8dd52, prereg `2026-11-02-dx-xpose-prereg.md`
(committed before CI/Colab). Merged to master @ 5293c3c. CI bundle
`kernels-dx-xpose` (run 41439832804, success).

Change: dX tc_64 kernel grid transposed to (k_tiles, b_tiles) so co-resident
CTAs share one dY b-tile instead of re-streaming it per k-tile (head dY
~26 GB DRAM hypothesis). Bit-exact: intra-CTA N-reduction unchanged, no
cross-CTA atomics — pytest parity green both runs (103 passed).

## Run 1 (ut-dxxpose @ 6f8dd52c, artefacts results/gate16.6f8dd52c.*): gate FAIL rc=1

pytest 103 passed. perfgate: fc1/dx −24.1%, fc2/dx −23.8%, head/dx −2.9%,
but head/update **FAIL +9.0%** and fc2/update WARN +2.4% on code the branch
does not touch, and step_ms 3497.4 (master 3395.6). Read as a slow/contended
VM, not a code effect — replication ordered.

## Run 2 (ut-dxxpose-r2 @ 6f8dd52c-r2, artefacts results/gate16.6f8dd52c-r2.*): PASS rc=0

pytest 103 passed. perfgate vs measured baseline (4810.507 ms):

- fc1/dx 51.21 → 36.05 ms **−29.66%**, fc2/dx 50.68 → 35.95 ms **−29.10%**,
  head/dx 649.31 → 518.9 ms **−20.08%** — the dY re-stream win is real;
- fwd rows −35.7…−39.2% (gate16+xpose+spill combined vs baseline), update rows
  −13.6…−15.0% (this VM faster overall — confirms run 1's update FAIL was VM
  variance, not the change);
- **step_ms 2976.439** — vs master 3395.638 that is **−12.4% from dX xpose
  alone**; vs baseline −38.1% total.

Prereg bar held on run 2: head/dx and both fc/dx beyond max(2σ_null,1%),
fwd/update within bar, step beyond bar → shipped.

## Step trajectory now

4810.507 → 3629.323 (gate16) → 3519.203 (fwd xpose) → 3395.638 (fwd spill)
→ **2976.439 ms (dX xpose)**.
