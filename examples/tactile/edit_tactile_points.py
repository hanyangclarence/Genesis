#!/usr/bin/env python3
"""Interactively visualize and remove tactile sensor points from a JSON file.

Usage:
    python examples/tactile/edit_tactile_points.py palm.json
    python examples/tactile/edit_tactile_points.py palm.json --link palm_link

Controls:
    Left-drag in any 2D panel  : lasso-select points (adds to selection)
    Delete / Backspace / D     : remove selected points
    Escape                     : clear selection
    U                          : undo last deletion
    S                          : save to file (overwrites input JSON)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath


def main():
    parser = argparse.ArgumentParser(description="Interactively remove tactile points from JSON")
    parser.add_argument("input", type=str, help="Path to tactile JSON file")
    parser.add_argument("--link", type=str, default=None,
                        help="Link name to edit (default: first link in file)")
    args = parser.parse_args()

    # Disable toolbar (pan/zoom mode intercepts left-drag before the lasso fires)
    plt.rcParams["toolbar"] = "None"
    # Clear conflicting key bindings
    plt.rcParams["keymap.save"] = []       # 's' → our save, not figure-save
    plt.rcParams["keymap.back"] = []       # backspace → our delete, not navigate-back
    plt.rcParams["keymap.quit"] = []       # don't let 'q' close the window unexpectedly

    path = Path(args.input)
    with open(path) as f:
        data = json.load(f)

    links = data.get("links", {})
    link_names = list(links.keys())
    if not link_names:
        print("No links found in JSON.")
        return

    link_name = args.link or link_names[0]
    if link_name not in links:
        print(f"Link '{link_name}' not found. Available links: {link_names}")
        return

    # Mutable state ---------------------------------------------------------
    state: dict = {
        "points": list(links[link_name]["points"]),
        "coords": np.empty((0, 3)),
        "selected": np.zeros(0, dtype=bool),
        "history": [],  # list of (points_copy, coords_copy) for undo
    }

    def _sync_coords():
        pts = state["points"]
        state["coords"] = np.array([p["local"] for p in pts]) if pts else np.empty((0, 3))
        state["selected"] = np.zeros(len(pts), dtype=bool)

    _sync_coords()
    print(f"Loaded {len(state['points'])} points for link '{link_name}'")

    # Figure layout ---------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    for ax, xlabel, ylabel, title in [
        (ax_xy, "X", "Y", "XY plane"),
        (ax_xz, "X", "Z", "XZ plane"),
        (ax_yz, "Y", "Z", "YZ plane"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")

    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("3D view (read-only)")

    c = state["coords"]
    sc_xy = ax_xy.scatter(c[:, 0], c[:, 1], s=8, c="steelblue", alpha=0.7)
    sc_xz = ax_xz.scatter(c[:, 0], c[:, 2], s=8, c="steelblue", alpha=0.7)
    sc_yz = ax_yz.scatter(c[:, 1], c[:, 2], s=8, c="steelblue", alpha=0.7)
    # 3D scatter is redrawn from scratch on each refresh (simpler than _offsets3d hack)

    def _refresh_3d():
        ax3d.cla()
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
        ax3d.set_title("3D view (read-only)")
        coords = state["coords"]
        if len(coords) == 0:
            return
        sel = state["selected"]
        colors = ["crimson" if s else "steelblue" for s in sel]
        sizes = np.where(sel, 25.0, 8.0)
        ax3d.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                     s=sizes, c=colors, alpha=0.7)  # type: ignore[arg-type]

    def _set_title(extra: str = ""):
        n = len(state["points"])
        n_sel = int(state["selected"].sum())
        msg = (
            f"{link_name}  |  {n} points  |  {n_sel} selected\n"
            "Lasso (2D panels) to select  ·  D/Delete to remove  ·  "
            "Esc to clear  ·  U to undo  ·  S to save"
        )
        fig.suptitle(extra if extra else msg, fontsize=10,
                     color="green" if extra else "black")

    def _refresh(extra: str = ""):
        coords = state["coords"]
        sel = state["selected"]
        colors = ["crimson" if s else "steelblue" for s in sel]
        sizes = [25 if s else 8 for s in sel]

        if len(coords):
            sc_xy.set_offsets(coords[:, :2])
            sc_xz.set_offsets(coords[:, [0, 2]])
            sc_yz.set_offsets(coords[:, 1:])
        else:
            sc_xy.set_offsets(np.empty((0, 2)))
            sc_xz.set_offsets(np.empty((0, 2)))
            sc_yz.set_offsets(np.empty((0, 2)))

        for sc in (sc_xy, sc_xz, sc_yz):
            sc.set_color(colors)
            sc.set_sizes(sizes)

        _refresh_3d()
        _set_title(extra)
        fig.canvas.draw_idle()

    _refresh()

    # Lasso selection -------------------------------------------------------

    def _make_onselect(proj: str):
        def onselect(verts):
            coords = state["coords"]
            if len(coords) == 0:
                return
            mpl_path = MplPath(verts)
            if proj == "xy":
                pts2d = coords[:, :2]
            elif proj == "xz":
                pts2d = coords[:, [0, 2]]
            else:
                pts2d = coords[:, 1:]
            newly = mpl_path.contains_points(pts2d)
            state["selected"] |= newly
            print(f"  [{proj}] lasso hit {int(newly.sum())} points "
                  f"({int(state['selected'].sum())} total selected)")
            _refresh()
        return onselect

    # Keep references alive so GC doesn't collect them
    _lasso_refs = [
        LassoSelector(ax_xy, _make_onselect("xy"), button=MouseButton.LEFT),
        LassoSelector(ax_xz, _make_onselect("xz"), button=MouseButton.LEFT),
        LassoSelector(ax_yz, _make_onselect("yz"), button=MouseButton.LEFT),
    ]

    # Key handler -----------------------------------------------------------

    def on_key(event):
        key = event.key

        if key in ("delete", "backspace", "d"):
            sel = state["selected"]
            if not sel.any():
                print("Nothing selected.")
                return
            n_remove = int(sel.sum())
            state["history"].append(
                (list(state["points"]), state["coords"].copy())
            )
            keep = ~sel
            state["points"] = [p for p, k in zip(state["points"], keep) if k]
            state["coords"] = state["coords"][keep]
            state["selected"] = np.zeros(len(state["points"]), dtype=bool)
            _refresh()
            print(f"Removed {n_remove} points. {len(state['points'])} remaining.")

        elif key == "escape":
            state["selected"][:] = False
            _refresh()

        elif key == "u":
            if not state["history"]:
                print("Nothing to undo.")
                return
            pts, coords = state["history"].pop()
            state["points"] = pts
            state["coords"] = coords
            state["selected"] = np.zeros(len(pts), dtype=bool)
            _refresh()
            print(f"Undone. {len(pts)} points.")

        elif key == "s":
            links[link_name]["points"] = state["points"]
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            n = len(state["points"])
            print(f"Saved {n} points to {path}")
            _refresh(extra=f"Saved!  {n} points written to {path}")

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Prevent _lasso_refs from being GC'd
    fig._lasso_refs = _lasso_refs  # type: ignore[attr-defined]

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
