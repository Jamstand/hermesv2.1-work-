# Nissan Silvia S15 Spec R for Roblox

A Nissan Silvia S15 Spec R built in Blender for Roblox: JDM right-hand drive, Lightning Yellow, stock 17" five-spoke wheels at factory ride height, factory Spec R aero, and a full SR20DET engine bay. Doors, bonnet and boot open, the lights work, and the parts are laid out for **A-Chassis**.

![front three-quarter](previews/front34.jpg)

| | |
|---|---|
| ![rear](previews/rear34.jpg) | ![side](previews/side.jpg) |
| ![doors, bonnet and boot open](previews/open.jpg) | ![SR20DET engine bay](previews/bay.jpg) |
| ![cabin](previews/cabin.jpg) | ![front](previews/front.jpg) |

## What's in the model

* **Body**: built to real S15 dimensions (4445 × 1695 × 1285 mm, 2525 mm wheelbase, 1470/1460 mm track). It's a hollow shell with a real cabin, engine bay and boot. It has panel gaps, the shoulder crease, side skirts, a fuel flap on the right rear quarter, a plate recess and a crease across the rear bumper.
* **Opening panels**: both doors, the bonnet and the boot are separate parts with hinge data. The door glass, mirrors, door cards and handles move with the doors. The rear wing, high-mount stop lamp and badges move with the boot lid.
* **Lights**: the headlights have chrome reflector bowls, a projector, a high beam and an amber indicator behind a clear lens. The tail lights have red tail/brake, amber indicator and clear reverse sections with chrome rings behind them. There are also fog lights, side repeaters, plate lamps, the wing brake light and dash needles, and each one is its own `Light_*` part.
* **Exterior**: diamond-mesh grilles (with the intercooler visible through the intake), a black front lip and rear valance, aero door mirrors, flap door handles with key locks, black B-pillars and window frames, wipers, the Spec R pedestal wing, a single exhaust on the left, JDM plates front and rear, and an "S" bonnet badge plus SILVIA and Spec R badges.
* **Interior (RHD)**: dashboard with the gauge binnacle and three dials, a boost gauge pod, centre stack with head unit, climate dials and vents, steering column and three-spoke wheel, pedals, console with gear lever and handbrake, front bucket seats with bolsters and inserts, the 2+2 rear bench, door cards, headliner, sun visors, rear-view mirror, floor mats, parcel-shelf speakers, and a spare wheel in the boot.
* **SR20DET bay**: red twin-cam valve cover with coil packs, intake plenum and runners, T28 turbo with downpipe, front-mount intercooler and piping, radiator, fan and hoses, air box, battery, fuse box, reservoirs, brake booster and master cylinder (driver's side), strut tops and a strut-tower bar, and a wiring loom.
* **Wheels**: 215/45R17 tyres with tread grooves and shoulder blocks, 17x7 five split-spoke rims with five lug nuts, a centre cap and a valve stem, plus discs and calipers.
* **Underbody**: suspension arms, coilovers, half shafts, the R200 diff, driveshaft, gearbox, fuel tank and the full exhaust run.

**Size:** 105 MeshParts and about 130k triangles. The largest single part has 16.6k, under Roblox's 20k-per-mesh cap. The full list is in [`export/parts.txt`](export/parts.txt).

## Files

| File | What it is |
|---|---|
| `export/S15_SpecR_Roblox.fbx` | **Import this into Roblox.** 1 unit = 1 stud, so the car is 15.9 studs long. |
| `export/S15_Setup.lua` | Command-bar script that sets up the imported car (see below). |
| `export/S15_SpecR.blend` | Editable Blender scene at real scale, in metres. |
| `export/S15_SpecR.glb` | glTF copy, in metres. |
| `export/parts.txt` | Every part with its assembly, material and triangle count. |
| `s15_spec_r.py` | The generator. Re-run it to rebuild everything. |

## Putting it in Roblox

1. **Import.** Go to *Home → Import 3D* and pick `S15_SpecR_Roblox.fbx`. Under *File Transform*, set **Scale Unit = Stud**, **World Forward = Front** and **World Up = Top**. These are Roblox's own recommended settings for Blender files. The car should come in about 15.9 studs long.
2. **Set up.** Select the imported model in the Explorer. Open *View → Command Bar*, paste in all of `export/S15_Setup.lua`, and press Enter. The script:
   * sets the material, colour, transparency and reflectance of every part;
   * builds the A-Chassis layout: `Body`, `Wheels/FL, FR, RL, RR` (each has an invisible cylinder wheel, with `Parts` for the tyre, rim and disc and `Fixed` for the caliper), `Misc/Door_L, Door_R, Hood, Trunk, SteeringWheel`, plus a right-hand-drive `DriveSeat` and a passenger `Seat`;
   * adds open/close prompts (key **F**) to both doors, the bonnet and the boot;
   * adds two server scripts (`S15_Panels`, `S15_Lights`), a `S15_LightEvent` RemoteEvent, and an A-Chassis plugin.
   It works out the car's orientation and scale from the four tyres, so it still works if the import was rotated or scaled.
3. **Chassis.** Drop your **A-Chassis Tune** into the new model, as you would with any A-Chassis car. Then move **`S15 Lights Plugin`** from the folder *"Move into A-Chassis Tune > Plugins"* into `A-Chassis Tune/Plugins`.
4. Everything is left anchored with CanCollide off, which is how A-Chassis expects a car before it initialises.

### Controls (from the plugin)

| Key | Action |
|---|---|
| L | Headlights, fogs, tail lights and dash lights |
| Z / C | Left / right indicator |
| X | Hazards |
| F (near a panel) | Open or close a door, the bonnet or the boot |

Brake lights, reverse lights and the steering-wheel rotation follow A-Chassis' `Values` (Brake, Gear and SteerC) automatically. Without a chassis, you can drive the lights from any script by setting these attributes on the car model: `Headlights`, `Brake`, `Reverse`, `IndicatorLeft`, `IndicatorRight`, `Hazards`.

On a parked, anchored display car the panels animate with a tween. On a running A-Chassis car they swing on servo `HingeConstraint`s.

## Changing it

* **Paint:** set `PAINT` near the top of `s15_spec_r.py`. The presets are `lightning_yellow`, `pearl_white`, `sparkling_silver`, `brilliant_blue`, `super_black` and `active_red`. You can also just change the `Body_Paint`, `Door_*_Paint`, `Hood_Paint` and `Trunk_Paint` colours in Studio.
* **Rebuild:** run `blender --background --python s15_spec_r.py` (or `-- --out <folder>`). In the Blender UI, use *Scripting → Open → Run Script*, which builds into a new `S15_SpecR` scene.
* **Preview renders:** run `blender -b -P s15_spec_r.py -- --no-export --render --open --samples 64`.

## Notes and limits

* The model is generated from code: lofted cross-sections plus exact booleans. It was shaped against the real dimensions and public reference photos of stock S15s; it isn't a scan. The overall proportions, the headlight and tail-light shapes, and the bumper openings, wing and stance are close to the real car, but small curvature details are approximations.
* I couldn't open Roblox Studio here. The generated Luau passes a type check against Roblox's API definitions (`luau-lsp`) and compiles, but it hasn't been run in Studio. A-Chassis builds also differ between versions. If yours already welds `Misc` or uses different `Values` names, adjust `S15_Panels` or the plugin to match.
* The calipers go in each wheel's `Fixed` model so they don't spin. If your A-Chassis version doesn't support `Fixed`, put them in `Parts` or weld them to the body.
* Stock S15 Spec Rs came with 16" or 17" wheels depending on the package. This one uses 17s (`TYRE_*` and `RIM_IN` in the script).

Reference dimensions: [auto-data.net](https://www.auto-data.net/en/nissan-silvia-s15-2.0-i-16v-t-250hp-automatic-24949), [carfromjapan.com](https://carfromjapan.com/specifications/nissan/silvia/581389f42afaa2c4b2869497), [supercars.net](https://www.supercars.net/blog/1999-nissan-silvia-spec-r/). Roblox import settings: [create.roblox.com/docs/art/blender](https://create.roblox.com/docs/art/blender).
