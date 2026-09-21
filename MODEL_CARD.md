# Model card

## Scope

This package estimates 2D and camera-space 3D locations for 12 predefined
human body regions from Gemini 2 L RGB-D frames. It also exposes experimental
standing-height and lying body-length estimates.

## Outputs

Each region can include a 2D center, 2D bounding box, 3D camera coordinate,
visibility probability, presence probability, confidence, depth validity, and
the source of the 3D estimate.

## Known limitations

- Public benchmark results may require a separate no-ground-truth-input audit.
- Gemini-domain evidence is currently limited and should not be interpreted as
  clinical accuracy.
- Heavy occlusion, missing feet/crown depth, multiple overlapping people, and
  unusual postures can cause rejection or increased error.
- The package does not contain an automated MRI-coil classifier.

## Intended use

Research, engineering validation, data collection, and human-in-the-loop
prototype development. It is not cleared for diagnosis, treatment, or clinical
decision-making.

