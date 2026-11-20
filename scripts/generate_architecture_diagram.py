from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_architecture_diagram(output_paths: list[Path]) -> None:
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_facecolor("#0F172A")  # Deep slate background
    fig.patch.set_facecolor("#0F172A")

    # Title Banner
    ax.text(
        0.5,
        0.96,
        "Dual-Stream R(2+1)D with 3D-CBAM Attention & TP-Pool Architecture",
        fontsize=20,
        fontweight="bold",
        color="#F8FAFC",
        ha="center",
        va="center",
    )
    ax.text(
        0.5,
        0.925,
        "End-to-End Inter-Frame Video Forgery Identification and Calibrated MS-SSIM Temporal Localization",
        fontsize=12,
        color="#94A3B8",
        ha="center",
        va="center",
    )

    def draw_box(
        ax, x, y, w, h, title, subtitle="", bg_color="#1E293B", border_color="#38BDF8", title_color="#F8FAFC"
    ):
        box = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            linewidth=2,
            edgecolor=border_color,
            facecolor=bg_color,
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x, y + 0.025, title, fontsize=11, fontweight="bold", color=title_color, ha="center", va="center")
            ax.text(x, y - 0.025, subtitle, fontsize=9, color="#CBD5E1", ha="center", va="center")
        else:
            ax.text(x, y, title, fontsize=11, fontweight="bold", color=title_color, ha="center", va="center")

    def draw_arrow(ax, x1, y1, x2, y2, label="", color="#38BDF8"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=15),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.015, my, label, fontsize=8, color="#94A3B8", ha="left", va="center")

    # 1. Input Box
    draw_box(
        ax,
        0.10,
        0.50,
        0.14,
        0.25,
        "Input Video Clip\n(T × H × W × 3)",
        "Raw Consecutive\nFrames K₁...K_T",
        bg_color="#1E293B",
        border_color="#60A5FA",
    )

    # Branching arrows
    draw_arrow(ax, 0.17, 0.56, 0.23, 0.72, "RGB")
    draw_arrow(ax, 0.17, 0.44, 0.23, 0.28, "Diff")

    # 2. Dual Stream Processing
    # Stream 1: RGB
    draw_box(
        ax,
        0.31,
        0.72,
        0.15,
        0.18,
        "Stream 1: Spatial RGB",
        "Raw Frame Appearance &\nCompression Discrepancies",
        bg_color="#0F2942",
        border_color="#38BDF8",
    )

    # Stream 2: Gradient Diff
    draw_box(
        ax,
        0.31,
        0.28,
        0.15,
        0.18,
        "Stream 2: Gradient Diff",
        "|K_f - K_{f+1}| × (1 + γ∇S_f)\nMotion Discontinuities",
        bg_color="#2E1065",
        border_color="#A855F7",
    )

    # Convergence arrows to Fusion
    draw_arrow(ax, 0.385, 0.72, 0.44, 0.56)
    draw_arrow(ax, 0.385, 0.28, 0.44, 0.44)

    # 3. Stream Fusion Block
    draw_box(
        ax,
        0.49,
        0.50,
        0.09,
        0.22,
        "Dual-Stream\nFusion",
        "Dynamic Gated\nConcatenation",
        bg_color="#1E1B4B",
        border_color="#818CF8",
    )

    draw_arrow(ax, 0.535, 0.50, 0.575, 0.50)

    # 4. R(2+1)D Backbone
    draw_box(
        ax,
        0.65,
        0.50,
        0.14,
        0.25,
        "R(2+1)D Backbone",
        "Spatial 2D Conv (1×3×3)\n+\nTemporal 1D Conv (3×1×1)",
        bg_color="#064E3B",
        border_color="#34D399",
    )

    draw_arrow(ax, 0.72, 0.50, 0.76, 0.50)

    # 5. 3D-CBAM Attention Block
    draw_box(
        ax,
        0.82,
        0.50,
        0.11,
        0.25,
        "3D-CBAM\nAttention",
        "Channel Attention\n+\nSpatial-Temporal Attn",
        bg_color="#4C1D95",
        border_color="#C084FC",
    )

    draw_arrow(ax, 0.875, 0.50, 0.915, 0.50)

    # 6. TP-Pool Block (Top Right)
    draw_box(
        ax,
        0.95,
        0.72,
        0.09,
        0.20,
        "TP-Pool",
        "F_peak || F_mean\nPreserves Seams",
        bg_color="#7C2D12",
        border_color="#FB923C",
    )

    draw_arrow(ax, 0.875, 0.56, 0.905, 0.72)

    # 7. Classification Output
    draw_box(
        ax,
        0.95,
        0.28,
        0.09,
        0.20,
        "Classifier Head",
        "Focal Loss FC\n[Auth / Insert / Delete]",
        bg_color="#831843",
        border_color="#F43F5E",
    )

    draw_arrow(ax, 0.875, 0.44, 0.905, 0.28)

    # 8. Localization Sub-Engine (Bottom Left to Right)
    loc_box = patches.FancyBboxPatch(
        (0.03, 0.03),
        0.94,
        0.12,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5,
        edgecolor="#F59E0B",
        facecolor="#1E1E10",
    )
    ax.add_patch(loc_box)

    ax.text(
        0.05,
        0.09,
        "Calibrated MS-SSIM Temporal Localization Sub-Engine:",
        fontsize=10,
        fontweight="bold",
        color="#FBBF24",
        va="center",
    )
    ax.text(
        0.05,
        0.05,
        "5-Scale Gaussian Pyramid MS-SSIM  →  Adaptive MAD Thresholding (τ_eff = median - 2.5·MAD)  →  Signature Dip Classifier  →  Boundary Indices [f_start, f_end]",
        fontsize=9,
        color="#FEF08A",
        va="center",
    )

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    plt.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved architecture diagram to: {path}")

    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        repo_root / "docs" / "figures" / "architecture_overview.png",
        repo_root / "architecture_overview.png",
        repo_root / "outputs" / "reports_enhanced" / "architecture_overview.png",
    ]
    create_architecture_diagram(paths)


if __name__ == "__main__":
    main()
