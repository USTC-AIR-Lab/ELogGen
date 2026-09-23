#!/usr/bin/env python3
"""Render a paged visual catalog of BH1K openable, fillable containers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from eloggen.simulation.omnigibson import activate, behavior_root, project_root


ROOT = project_root()
ASSET_ROOT = behavior_root() / "datasets/behavior-1k-assets/objects"
PROPERTIES = behavior_root() / "bddl/bddl/generated_data/properties_to_synsets.json"
DEFAULT_OUTPUT = ROOT / "artifacts/openable_container_catalog"

activate()
CABINET_CATEGORIES = (
    "bottom_cabinet",
    "bottom_cabinet_no_top",
    "metal_bottom_cabinet",
    "top_cabinet",
    "locker",
    "cedar_chest",
    "toolbox",
)


def discover_assets(scope: str) -> list[tuple[str, str]]:
    if scope == "cabinets":
        categories = CABINET_CATEGORIES
    else:
        from bddl.object_taxonomy import ObjectTaxonomy

        props = json.loads(PROPERTIES.read_text(encoding="utf-8"))
        synsets = sorted(set(props["openable"]) & set(props["fillable"]))
        taxonomy = ObjectTaxonomy()
        categories = sorted({category for synset in synsets for category in taxonomy.get_subtree_categories(synset)})
    return [
        (category, model_dir.name)
        for category in categories
        if (ASSET_ROOT / category).is_dir()
        for model_dir in sorted((ASSET_ROOT / category).iterdir())
        if model_dir.is_dir() and (model_dir / "misc/metadata.json").is_file()
    ]


def render_assets(assets: list[tuple[str, str]], output: Path) -> list[dict]:
    import torch as th
    from PIL import Image
    import omnigibson as og
    import omnigibson.utils.transform_utils as T
    from omnigibson.macros import gm
    from omnigibson.objects import DatasetObject

    gm.HEADLESS = True
    output.mkdir(parents=True, exist_ok=True)
    cfg = {
        "scene": {"type": "Scene"},
        "objects": [
            {"type": "LightObject", "light_type": "Sphere", "name": "key", "radius": 0.01,
             "intensity": 8e4, "position": [-2.0, -2.0, 3.0]},
            {"type": "LightObject", "light_type": "Sphere", "name": "fill", "radius": 0.01,
             "intensity": 5e4, "position": [2.0, -1.0, 2.0]},
        ],
    }
    env = og.Environment(configs=cfg)
    og.sim.viewer_width = 640
    og.sim.viewer_height = 480
    camera_position = th.tensor([0.0, -2.1, 1.35])
    camera_orientation = th.tensor([0.6350064, 0.0, 0.0, 0.77250687])
    og.sim.viewer_camera.set_position_orientation(camera_position, camera_orientation)
    records = []

    for index, (category, model) in enumerate(assets):
        name = f"catalog_object_{index}"
        obj = DatasetObject(name=name, category=category, model=model, visual_only=True, position=[0.0, 0.0, 5.0])
        env.scene.add_object(obj)
        env.step(th.empty(0))
        native_extent = obj.aabb_extent.clone()
        og.sim.stop()
        obj.scale = (th.ones(3) * 0.9 / native_extent).min()
        og.sim.play()
        env.step(th.empty(0))
        center_offset = obj.get_position_orientation()[0] - obj.aabb_center + th.tensor([0.0, 0.0, 1.0])
        joint_types = sorted({str(joint.joint_type).replace("Joint", "") for joint in obj.joints.values()})
        handle_links = sorted(name for name in obj.links if "handle" in name.lower())
        views = []
        for view_index, yaw in enumerate((-math.pi / 2, -math.pi / 4)):
            quat = T.euler2quat(th.tensor([0.0, 0.0, yaw]))
            obj.set_position_orientation(position=center_offset, orientation=quat)
            for _ in range(3):
                env.step(th.empty(0))
            rgb = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3].cpu().numpy()
            view_path = output / f"{index:03d}_{category}_{model}_v{view_index}.png"
            Image.fromarray(rgb).save(view_path)
            views.append(str(view_path))
        record = {
            "index": index,
            "category": category,
            "model": model,
            "native_extent_m": [round(float(value), 4) for value in native_extent],
            "joint_types": joint_types,
            "joint_count": len(obj.joints),
            "handle_links": handle_links,
            "views": views,
        }
        records.append(record)
        print(f"[{index + 1}/{len(assets)}] {category}/{model}: {joint_types}, handles={len(handle_links)}", flush=True)
        env.scene.remove_object(obj=obj)
    return records


def make_contact_sheets(records: list[dict], output: Path, page_size: int = 12) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    card_w, card_h = 640, 300
    for page_start in range(0, len(records), page_size):
        page_records = records[page_start : page_start + page_size]
        sheet = Image.new("RGB", (card_w * 3, card_h * 4), "#182028")
        draw = ImageDraw.Draw(sheet)
        for slot, record in enumerate(page_records):
            x, y = (slot % 3) * card_w, (slot // 3) * card_h
            left = Image.open(record["views"][0]).convert("RGB").resize((320, 240))
            right = Image.open(record["views"][1]).convert("RGB").resize((320, 240))
            sheet.paste(left, (x, y))
            sheet.paste(right, (x + 320, y))
            extent = " x ".join(f"{value:.2f}" for value in record["native_extent_m"])
            joints = ",".join(record["joint_types"]) or "none"
            label = (
                f"#{record['index']:03d}  {record['category']}/{record['model']}\n"
                f"size {extent} m | joints {joints} | handles {len(record['handle_links'])}"
            )
            draw.rectangle((x, y + 240, x + card_w, y + card_h), fill="#10161c")
            draw.multiline_text((x + 10, y + 248), label, fill="white", font=font, spacing=5)
        page = page_start // page_size + 1
        sheet.save(output / f"catalog_page_{page:02d}.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("cabinets", "all"), default="cabinets")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    assets = discover_assets(args.scope)
    if args.limit is not None:
        assets = assets[: args.limit]
    records = render_assets(assets, args.output)
    (args.output / "catalog.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    make_contact_sheets(records, args.output)
    prismatic = [record for record in records if "Prismatic" in record["joint_types"]]
    priority_output = args.output / "prismatic_candidates"
    priority_output.mkdir(exist_ok=True)
    make_contact_sheets(prismatic, priority_output)
    (priority_output / "catalog.json").write_text(json.dumps(prismatic, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} assets and contact sheets to {args.output}")


if __name__ == "__main__":
    main()
