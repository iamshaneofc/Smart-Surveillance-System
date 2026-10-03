# Annotation format

One JSON file per manifest entry, referenced by `entry.annotation` or defaulting
to `annotations/<entry_id>.json`.

```json
{
  "entry_id": "example-001",
  "frames": [
    {
      "frame_index": 0,
      "boxes": [
        {"class_name": "person", "bbox": [0.42, 0.31, 0.15, 0.55]}
      ]
    }
  ]
}
```

- `bbox` is normalized `[x, y, width, height]` in `[0, 1]` (frame-relative),
  matching the coordinates used by zone polygons.
- For videos, `frame_index` selects which decoded frames are evaluated; only
  annotated frames are scored.
- `confidence` on a box is ignored (ground truth has no confidence).
