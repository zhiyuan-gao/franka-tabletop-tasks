# Runtime asset provenance

The calibration files below were copied byte-for-byte from the immutable
Track B handoff snapshot on 2026-08-27. The source snapshot is not required at
runtime; all referenced files are stored under this project's `assets/`
directory.

| Project-local file | SHA-256 |
|---|---|
| `calibration/background/empty_background.png` | `3ccfd765f3a85185c5f253f7c79bf3468bf2be5892cdc5ddf18383787db79bbe` |
| `calibration/configs/simulation_camera.yaml` | `5e7c135753d4e73603475a5c85ef0dd1294d7d8b5e16e1f8794aefc1f2217425` |
| `calibration/transforms/T_base_from_camera_usd.npy` | `07326e3d97239a82b7e14a8cb489d7b86c956f9b9631153608f40bcc40967032` |
| `calibration/transforms/T_base_from_marker.npy` | `5df618930314c0f22ced96573177c6c29fac1c41293cc2821b85ccc0b96865b8` |
| `task_assets/cabinets/SingleDoorWhiteCabinet.usda` | `046c5f1e0b1f4697b79ab72965ae509ece176997fa90124d72504b0b1e980d7e` |
| `task_assets/cabinets/TwoDrawerWhiteCabinet.usda` | `354d5351119ee22aa563541b84fe6bf0789d95bca623d86508b4fed77cab93dc` |
| `task_assets/pouring/BluePouringCup.usda` | `a18e3d6c209e4a1320cc96e9ff6b266d8c0f4bac1bb62f48583ab7d8bb5f778a` |
| `task_assets/pouring/PurplePouringBowl.usda` | `b5d07934b9ab8990a485a878f8d2eadb2245d417ba35eb83fe0cb3c54b45bf2a` |
| `task_assets/ycb/011_banana.usd` | `d4d1fa7af05967365cd538824ac68401111c574865a7fba7bc76d861f54f25dd` |
| `task_assets/ycb/011_banana_physics.usda` | `f56dc39388005f0d5a539fca8039363cfa2b915e39a1f4197428d683494917ff` |
| `task_assets/ycb/Materials/Textures/011_banana_COLOR.png` | `841dd1e67d23815cfaac3a14019c4dd8712d956664dab853a4d558353eba3795` |

PushObjectToRegion loads `004_sugar_box.usd` from the official Isaac 4.5 YCB
Axis_Aligned_Physics asset URL. It is not copied from LW-BenchHub. The box is
spawned at 0.8 scale with 0.08 kg mass and fixed 0.50/0.40 static/dynamic
friction for the initial easy-push baseline.

The landscape A4 target sheet is aligned over the measured 0.268 x 0.190 m
ChArUco board using `T_base_from_marker.npy`. Its 18 x 12 cm printed blue rectangle is
centered on the sheet; neither element is an additional fiducial marker.

The original `011_banana.usd` is byte-identical to NVIDIA's Isaac 4.5 YCB
`Axis_Aligned` visual asset. The project-local `011_banana_physics.usda` is a
non-destructive wrapper around it, adding a 0.066 kg PhysX rigid body, bounded
convex decomposition, collision material, and solver properties. It keeps the
official visual and texture files unchanged.
