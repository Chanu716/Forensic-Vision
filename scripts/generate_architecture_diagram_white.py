from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_white_architecture_diagram(output_paths: list[Path]) -> None:
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(16, 8.5), dpi=300)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    def draw_dashed_container(ax, x, y, w, h, title, title_color="#0F172A", bg_color="#FAFAFA"):
        box = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.5,
            linestyle="--",
            edgecolor="#64748B",
            facecolor=bg_color,
        )
        ax.add_patch(box)
        ax.text(x - w / 2 + 0.02, y + h / 2 - 0.035, title, fontsize=13, fontweight="bold", color=title_color, ha="left", va="top")

    def draw_block(ax, x, y, w, h, title, subtitle="", bg_color="#FFFFFF", border_color="#000000", text_color="#000000"):
        box = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=border_color,
            facecolor=bg_color,
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x, y + 0.018, title, fontsize=10, fontweight="bold", color=text_color, ha="center", va="center")
            ax.text(x, y - 0.018, subtitle, fontsize=8.5, color="#334155", ha="center", va="center")
        else:
            ax.text(x, y, title, fontsize=10, fontweight="bold", color=text_color, ha="center", va="center")

    def draw_arrow(ax, x1, y1, x2, y2, color="#000000"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=12),
        )

    # 1. Input Box (Left)
    ax.text(0.06, 0.60, "Input Video Clip", fontsize=11, fontweight="bold", ha="center", va="center")
    ax.text(0.06, 0.56, r"$(T \times H \times W \times 3)$", fontsize=9.5, style="italic", ha="center", va="center")
    
    # Simple stack icon representation
    for i in range(3):
        stack_box = patches.Rectangle((0.02 + i*0.01, 0.40 - i*0.01), 0.06, 0.12, linewidth=1, edgecolor="#334155", facecolor="#E2E8F0")
        ax.add_patch(stack_box)
    ax.text(0.05, 0.44, "...", fontsize=14, fontweight="bold", ha="center", va="center")

    # Arrows from Input to Streams
    draw_arrow(ax, 0.10, 0.50, 0.14, 0.50)
    draw_arrow(ax, 0.14, 0.50, 0.17, 0.72)
    draw_arrow(ax, 0.14, 0.50, 0.17, 0.28)

    # 2. RGB Appearance Stream Container (Top)
    draw_dashed_container(ax, 0.43, 0.72, 0.51, 0.38, "RGB Appearance Stream", title_color="#0F172A", bg_color="#FFFBEB")

    # Inside RGB Stream
    ax.text(0.23, 0.83, "RGB Frames", fontsize=9.5, fontweight="bold")
    ax.text(0.23, 0.79, r"$(T \times H \times W \times 3)$", fontsize=8.5, style="italic")
    for i in range(3):
        box = patches.Rectangle((0.21 + i*0.008, 0.65 - i*0.008), 0.04, 0.10, linewidth=1, edgecolor="#2563EB", facecolor="#DBEAFE")
        ax.add_patch(box)

    draw_arrow(ax, 0.27, 0.72, 0.30, 0.72)

    # R(2+1)D Blocks x4
    draw_block(ax, 0.35, 0.72, 0.09, 0.18, "R(2+1)D\nResidual\nBlocks", "×4", bg_color="#FEE2E2", border_color="#EF4444")
    draw_arrow(ax, 0.395, 0.72, 0.425, 0.72)

    # 3D-CBAM Attention
    draw_block(ax, 0.47, 0.72, 0.08, 0.18, "3D-CBAM\nAttention", "", bg_color="#DBEAFE", border_color="#3B82F6")
    draw_arrow(ax, 0.51, 0.72, 0.54, 0.72)

    # TP-Pool
    draw_block(ax, 0.59, 0.72, 0.09, 0.18, "Temporal\nPeak-Preserving\nPooling\n(TP-Pool)", "", bg_color="#DCFCE7", border_color="#22C55E")
    draw_arrow(ax, 0.635, 0.72, 0.665, 0.72)

    # f_rgb vector
    f_rgb_box = patches.Rectangle((0.665, 0.65), 0.015, 0.14, linewidth=1, edgecolor="#1E40AF", facecolor="#93C5FD")
    ax.add_patch(f_rgb_box)
    ax.text(0.672, 0.82, r"$f_{rgb}$", fontsize=11, fontweight="bold", style="italic", ha="center")
    ax.text(0.672, 0.61, "(D)", fontsize=9, ha="center")

    # 3. Temporal Difference Stream Container (Bottom)
    draw_dashed_container(ax, 0.43, 0.28, 0.51, 0.38, "Temporal Difference Stream", title_color="#0F172A", bg_color="#F0F9FF")

    # Inside Diff Stream
    ax.text(0.23, 0.39, "Frame Difference", fontsize=9.5, fontweight="bold")
    ax.text(0.23, 0.35, r"$(T-1 \times H \times W \times 1)$", fontsize=8.5, style="italic")
    for i in range(3):
        box = patches.Rectangle((0.21 + i*0.008, 0.21 - i*0.008), 0.04, 0.10, linewidth=1, edgecolor="#475569", facecolor="#94A3B8")
        ax.add_patch(box)

    draw_arrow(ax, 0.27, 0.28, 0.30, 0.28)

    # R(2+1)D Blocks x4
    draw_block(ax, 0.35, 0.28, 0.09, 0.18, "R(2+1)D\nResidual\nBlocks", "×4", bg_color="#FEE2E2", border_color="#EF4444")
    draw_arrow(ax, 0.395, 0.28, 0.425, 0.28)

    # 3D-CBAM Attention
    draw_block(ax, 0.47, 0.28, 0.08, 0.18, "3D-CBAM\nAttention", "", bg_color="#DBEAFE", border_color="#3B82F6")
    draw_arrow(ax, 0.51, 0.28, 0.54, 0.28)

    # TP-Pool
    draw_block(ax, 0.59, 0.28, 0.09, 0.18, "Temporal\nPeak-Preserving\nPooling\n(TP-Pool)", "", bg_color="#DCFCE7", border_color="#22C55E")
    draw_arrow(ax, 0.635, 0.28, 0.665, 0.28)

    # f_diff vector
    f_diff_box = patches.Rectangle((0.665, 0.21), 0.015, 0.14, linewidth=1, edgecolor="#1E40AF", facecolor="#93C5FD")
    ax.add_patch(f_diff_box)
    ax.text(0.672, 0.38, r"$f_{diff}$", fontsize=11, fontweight="bold", style="italic", ha="center")
    ax.text(0.672, 0.17, "(D)", fontsize=9, ha="center")

    # Output arrows to Gated Feature Fusion
    draw_arrow(ax, 0.68, 0.72, 0.72, 0.55)
    draw_arrow(ax, 0.68, 0.28, 0.72, 0.45)

    # 4. Gated Feature Fusion Container
    draw_dashed_container(ax, 0.83, 0.50, 0.24, 0.30, "Gated Feature Fusion", title_color="#0F172A", bg_color="#FAF5FF")

    # Inside Gated Fusion
    draw_block(ax, 0.77, 0.50, 0.07, 0.14, r"$[f_{rgb} \,\|\, f_{diff}]$", "Concatenation\n(2D)", bg_color="#E2E8F0", border_color="#64748B")
    draw_arrow(ax, 0.805, 0.50, 0.835, 0.50)

    draw_block(ax, 0.87, 0.50, 0.07, 0.14, "Gating\nMechanism", "", bg_color="#F3E8FF", border_color="#A855F7")
    draw_arrow(ax, 0.905, 0.50, 0.935, 0.50)

    # f_fused vector
    f_fused_box = patches.Rectangle((0.935, 0.43), 0.015, 0.14, linewidth=1, edgecolor="#6B21A8", facecolor="#E9D5FF")
    ax.add_patch(f_fused_box)
    ax.text(0.942, 0.60, r"$f_{fused}$", fontsize=11, fontweight="bold", style="italic", ha="center")
    ax.text(0.942, 0.39, "(D)", fontsize=9, ha="center")

    # Arrow from Fusion down to Classification Head
    draw_arrow(ax, 0.83, 0.35, 0.83, 0.28)

    # 5. Classification Head Container (Bottom Right)
    draw_dashed_container(ax, 0.83, 0.15, 0.24, 0.22, "Classification Head", title_color="#0F172A", bg_color="#FFF7ED")

    draw_block(ax, 0.75, 0.14, 0.06, 0.11, "Fully\nConnected", "", bg_color="#FEF3C7", border_color="#D97706")
    draw_arrow(ax, 0.78, 0.14, 0.80, 0.14)

    draw_block(ax, 0.83, 0.14, 0.05, 0.11, "Softmax", "", bg_color="#FEE2E2", border_color="#EF4444")
    draw_arrow(ax, 0.855, 0.14, 0.875, 0.14)

    # Output classes
    draw_block(ax, 0.91, 0.20, 0.05, 0.05, "Authentic", "", bg_color="#F1F5F9", border_color="#475569")
    draw_block(ax, 0.91, 0.14, 0.05, 0.05, "Frame Insertion", "", bg_color="#F1F5F9", border_color="#475569")
    draw_block(ax, 0.91, 0.08, 0.05, 0.05, "Frame Deletion", "", bg_color="#F1F5F9", border_color="#475569")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.95)
    ax.axis("off")

    plt.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved publication-quality white architecture diagram to: {path}")

    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        repo_root / "docs" / "figures" / "architecture_overview.png",
        repo_root / "architecture_overview.png",
        repo_root / "outputs" / "reports_enhanced" / "architecture_overview.png",
    ]
    create_white_architecture_diagram(paths)


if __name__ == "__main__":
    main()
