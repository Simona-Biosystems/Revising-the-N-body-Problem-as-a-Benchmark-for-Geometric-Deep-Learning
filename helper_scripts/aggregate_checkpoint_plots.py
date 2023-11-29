import argparse
import glob
import os

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont


def create_single_pdf_from_all_plots(
    checkpoints_folder, plot_names, max_checkpoint=None, single_checkpoint=None
):
    checkpoint_dirs = sorted(
        [
            d
            for d in glob.glob(os.path.join(checkpoints_folder, "*"))
            if os.path.basename(d).isdigit()
        ],
        key=lambda x: int(os.path.basename(x)),
    )

    if single_checkpoint is not None:
        checkpoint_dirs = [
            d for d in checkpoint_dirs if int(os.path.basename(d)) == single_checkpoint
        ]
    elif max_checkpoint:
        checkpoint_dirs = [
            d for d in checkpoint_dirs if int(os.path.basename(d)) <= max_checkpoint
        ]

    print(checkpoint_dirs)
    pdf_output_path = os.path.join(checkpoints_folder, f"all_plots_summary.pdf")

    # Create a new PDF to store the merged output
    merged_pdf = fitz.open()

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_number = os.path.basename(checkpoint_dir)
        images = []
        for plot_name in plot_names:
            plot_files = sorted(
                glob.glob(
                    os.path.join(
                        checkpoint_dir,
                        "generated_trajectories",
                        "*",
                        "plots",
                        plot_name,
                    )
                )
            )
            for plot_file in plot_files:
                if plot_file.endswith(".png"):
                    image = Image.open(plot_file)
                    images.append(image)
                elif (
                    plot_file.endswith(".pdf")
                    and "trajectories_static.pdf" in plot_file
                ):
                    with fitz.open(plot_file) as pdf_doc:
                        if pdf_doc.page_count >= 2:
                            page1 = pdf_doc.load_page(0)
                            page2 = pdf_doc.load_page(1)
                            pix1 = page1.get_pixmap()
                            pix2 = page2.get_pixmap()
                            image1 = Image.frombytes(
                                "RGB", [pix1.width, pix1.height], pix1.samples
                            )
                            image2 = Image.frombytes(
                                "RGB", [pix2.width, pix2.height], pix2.samples
                            )
                            combined_height = image1.height + image2.height
                            combined_width = max(image1.width, image2.width)
                            combined_image = Image.new(
                                "RGB", (combined_width, combined_height)
                            )
                            combined_image.paste(image1, (0, 0))
                            combined_image.paste(image2, (0, image1.height))
                            images.append(combined_image)

        if images:
            # Resize all images to have the same height
            target_height = max(im.size[1] for im in images)
            resized_images = []
            for im in images:
                ratio = target_height / im.size[1]
                new_width = int(im.size[0] * ratio)
                resized_im = im.resize((new_width, target_height))
                resized_images.append(resized_im)

            # Get total width of all resized images
            total_width = sum(im.size[0] for im in resized_images)

            # Add space between rows (checkpoints)
            margin_between_rows = 20  # Adjust the vertical margin between checkpoints

            # Create combined image including the margin
            combined_image = Image.new(
                "RGB", (total_width, target_height + margin_between_rows), "white"
            )

            x_offset = 0
            for im in resized_images:
                combined_image.paste(im, (x_offset, 0))
                x_offset += im.size[0]

            # Optional: Add checkpoint number text
            draw = ImageDraw.Draw(combined_image)
            try:
                # Try to use a TrueType font
                font = ImageFont.truetype("arial.ttf", size=36)
            except IOError:
                # Fall back to the default font if Arial is not available
                font = ImageFont.load_default().font_variant(size=36)
            draw.text(
                (10, 10), f"Checkpoint {checkpoint_number}", fill="black", font=font
            )

            # Save to a temporary PDF
            pdf_temp_path = os.path.join(checkpoint_dir, "combined_temp.pdf")
            combined_image.save(pdf_temp_path, "PDF", resolution=100.0)
            with fitz.open(pdf_temp_path) as doc:
                merged_pdf.insert_pdf(doc)
            os.remove(pdf_temp_path)

    # Optionally, apply compression
    merged_pdf.save(pdf_output_path, deflate=True)
    merged_pdf.close()

    print(f"Saved: {pdf_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a single PDF from all plots in checkpoint directories."
    )
    parser.add_argument("--folder", type=str, help="Path to the checkpoints folder")
    parser.add_argument(
        "--max-checkpoint",
        type=int,
        default=None,
        help="Maximum checkpoint number to include (default: all)",
    )
    parser.add_argument(
        "--single-checkpoint",
        type=int,
        default=None,
        help="Process only a single checkpoint number",
    )
    args = parser.parse_args()

    plot_names = [
        "collision_distribution.png",
        "sticking_distribution.png",
        "group_collision_distribution_multiplot.png",
        "leaving_distribution.png",
        "sharp_turns_distribution_30.png",
        "sharp_turns_distribution_45.png",
        "energies.png",
        "trajectories_static.pdf",  # Include the PDF plot name
    ]  # Array of plot names

    create_single_pdf_from_all_plots(
        args.folder, plot_names, args.max_checkpoint, args.single_checkpoint
    )
