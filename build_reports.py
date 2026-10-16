"""Builds one PDF report per unit:  python build_reports.py [unit numbers...]"""
import os
import sys
from reportlab.lib.units import mm
from reportlab.platypus import Spacer, PageBreak, KeepTogether
from report_engine import (para, steps, bullets, image, table, heading, code_block, terminal_png, build,
                           CONTENT_W)

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
os.makedirs(REPORTS, exist_ok=True)


def p(*parts):
    return os.path.join(ROOT, *parts)


def console(folder, script):
    txt = open(p(folder, "outputs", "console.txt")).read().replace(ROOT + "/", "")
    return terminal_png(txt, f"python {script}", p(folder, "outputs", "console_screenshot.png"))


def standard(unit_no, title, task, aim, algo, folder, script, inputs, outputs, tables, observations, conclusion,
             filename):
    st = [heading(1, "Aim"), para(aim), Spacer(1, 4)]
    st.append(heading(2, "Algorithm / Procedure"))
    for sub, items in algo:
        if sub:
            st.append(para(sub, "h2"))
        st += steps(items)
    st += [PageBreak(), heading(3, "Python Code")] + code_block(p(folder, script))
    st += [PageBreak(), heading(4, "Input Image")]
    st += [image(path, cap, **kw) for path, cap, kw in inputs]
    st += [PageBreak(), heading(5, "Output Screenshots"),
           image(console(folder, script), "Program output in the terminal", max_h=110 * mm)]
    st += [image(path, cap, **kw) for path, cap, kw in outputs]
    st += [PageBreak(), heading(6, "Result Analysis / Observations")]
    for t in tables:
        st += [t, Spacer(1, 12)]
    st += bullets(observations)
    st += [Spacer(1, 6), KeepTogether([heading(7, "Conclusion"), para(conclusion)])]
    build(os.path.join(REPORTS, filename), unit_no, title, task, st)


# =====================================================================
def unit1():
    F = "Unit1_Filtering"
    o = lambda f: p(F, "outputs", f)
    standard(
        1, "Image Filtering",
        "Implement custom spatial domain filtering. Load an image, introduce artificial noise (Salt-and-Pepper / "
        "Gaussian), and apply Median, Mean, and Gaussian Filters to compare their noise-reduction and image-blurring "
        "efficiencies.",
        "To write custom Mean, Median and Gaussian filters, apply them to an image with Salt-and-Pepper and Gaussian "
        "noise, and compare how well each filter removes the noise and how much it blurs the image.",
        [("", [
            "Load the grayscale Cameraman image (512 x 512).",
            "Add Salt-and-Pepper noise (5% of pixels) and Gaussian noise (mean 0, σ = 20).",
            "For every pixel, take its 3 x 3 neighbourhood (sliding window).",
            "Mean filter: replace the pixel with the average of the neighbourhood.",
            "Median filter: replace the pixel with the median of the neighbourhood.",
            "Gaussian filter: replace the pixel with a weighted average using a Gaussian kernel (σ = 1).",
            "Compare each output with the original image using PSNR and SSIM.",
            "Measure blurring using the variance of the Laplacian (lower value = more blur).",
        ])],
        F, "unit1_filtering.py",
        [(o("input.png"), "Input image (Cameraman, 512 x 512). Source: scikit-image sample data", dict(max_h=95 * mm)),
         (o("fig1_noisy.png"), "Original image and images with added noise", {})],
        [(o("fig2_filtered.png"), "Filter outputs for Salt-and-Pepper noise", {}),
         (o("fig3_filtered.png"), "Filter outputs for Gaussian noise", {}),
         (o("fig4_zoom.png"), "Zoomed view of the filtered images", {}),
         (o("fig5_psnr_chart.png"), "PSNR comparison of the three filters", dict(width=CONTENT_W * 0.9))],
        [table([["Noise type", "Filter", "PSNR (dB)", "SSIM"],
                ["Salt & Pepper", "No filter", "17.77", "0.358"], ["", "Mean", "24.88", "0.507"],
                ["", "Median", "30.11", "0.864"], ["", "Gaussian", "24.94", "0.520"],
                ["Gaussian", "No filter", "22.41", "0.367"], ["", "Mean", "27.39", "0.641"],
                ["", "Median", "26.94", "0.575"], ["", "Gaussian", "27.94", "0.653"]],
               [38 * mm, 34 * mm, 30 * mm, 26 * mm]),
         table([["Kernel size", "Mean", "Median", "Gaussian"],
                ["3 x 3", "79.9", "319.9", "88.9"], ["5 x 5", "22.9", "194.0", "23.9"],
                ["7 x 7", "9.5", "138.3", "9.4"]],
               [32 * mm, 32 * mm, 32 * mm, 32 * mm])],
        ["The first table shows noise removal. Higher PSNR and SSIM mean the result is closer to the original image.",
         "For Salt-and-Pepper noise, the Median filter gave the best result (PSNR 30.11 dB). Mean and Gaussian filters "
         "only spread the white and black dots into grey spots.",
         "For Gaussian noise, the Gaussian filter gave the best result (27.94 dB), with the Mean filter very close.",
         "The second table shows sharpness (original image = 1133.2). Mean and Gaussian filters blur the edges much "
         "more than the Median filter, and the blur increases with the kernel size.",
         "The custom filters gave the same output as the OpenCV functions."],
        "The Median filter is the best choice for Salt-and-Pepper noise because it removes the noise and keeps the "
        "edges sharp. The Gaussian filter works best for Gaussian noise. Mean and Gaussian filters blur the image "
        "more as the kernel size grows, so a small 3 x 3 kernel gives a good balance between noise removal and detail.",
        "Unit1_Image_Filtering.pdf")


# =====================================================================
def unit2():
    F = "Unit2_Edge_Point_Detection"
    o = lambda f: p(F, "outputs", f)
    standard(
        2, "Edge & Point Detection",
        "Multi-stage boundary extraction. Implement Canny Edge Detection and a Harris Corner Detector on a complex "
        "structural image (like a checkerboard or architectural building) to analyze local gradient intensities and "
        "highlight key interest points.",
        "To detect edges using the Canny edge detector and corner points using the Harris corner detector on a "
        "building image and a checkerboard image.",
        [("A. Canny Edge Detection", [
            "Convert the image to grayscale.",
            "Smooth the image with a 5 x 5 Gaussian filter to reduce noise.",
            "Find the gradients Gx and Gy using the Sobel operator, then the gradient magnitude and direction.",
            "Non-maximum suppression: keep only the pixels that are a local maximum along the gradient direction.",
            "Double threshold: mark pixels as strong edges, weak edges or non-edges.",
            "Hysteresis: keep a weak edge only if it is connected to a strong edge.",
        ]), ("B. Harris Corner Detection", [
            "Find the image derivatives Ix and Iy.",
            "Compute Ix², Iy² and IxIy and smooth them with a Gaussian window.",
            "Compute the corner response R = det(M) - k (trace M)² with k = 0.04.",
            "Keep the points where R is above a threshold and is a local maximum.",
            "Mark the corners on the image.",
        ])],
        F, "unit2_edge_corner.py",
        [(p(F, "building.jpg"), "Input image 1 - building with a grid of windows. Source: Wikimedia Commons, by Wilfredor, CC BY-SA 4.0", {}),
         (o("input_checkerboard.png"), "Input image 2 - checkerboard with perspective distortion (generated in the program)",
          dict(max_h=85 * mm))],
        [(o("building_canny_stages.png"), "Stages of Canny edge detection - building", dict(max_h=150 * mm)),
         (o("building_results.png"), "Building - Canny edges, Harris response and corners", dict(max_h=200 * mm)),
         (o("checkerboard_canny_stages.png"), "Stages of Canny edge detection - checkerboard", {}),
         (o("checkerboard_results.png"), "Checkerboard - Canny edges, Harris response and corners", {})],
        [table([["Result", "Building", "Checkerboard"],
                ["Mean gradient magnitude", "60.7", "38.1"],
                ["Edge pixels after NMS", "57,052", "5,721"],
                ["Edge pixels after hysteresis", "26,224", "5,721"],
                ["Match with OpenCV Canny", "98.89 %", "99.83 %"],
                ["Harris corners found", "434", "81"]],
               [60 * mm, 40 * mm, 40 * mm])],
        ["The building has a higher mean gradient because the brick wall has a lot of texture.",
         "Hysteresis removed more than half of the edge pixels in the building image. These were mostly weak edges "
         "from the bricks. The window frames were kept.",
         "All 81 corners of the checkerboard (9 x 9 grid) were found, even with the perspective distortion.",
         "In the building image, the corners were found mostly at the window corners after light smoothing.",
         "The Harris response is high at corners, negative along edges and close to zero in flat areas."],
        "Canny edge detection gave thin and continuous edges, and its multi-stage process removed most of the texture "
        "noise. The Harris detector found the important corner points in both images. Edges have a strong gradient in "
        "one direction, while corners have strong gradients in two directions.",
        "Unit2_Edge_and_Point_Detection.pdf")


# =====================================================================
def unit3():
    F = "Unit3_Binary_Shape_Analysis"
    o = lambda f: p(F, "outputs", f)
    standard(
        3, "Binary Shape Analysis",
        "Write a Python script to perform Mathematical Morphology. Read a silhouette image, perform Erosion, Dilation, "
        "Opening, and Closing, and extract the structural 8-directional Chain Code representation of the isolated "
        "binary boundary.",
        "To perform Erosion, Dilation, Opening and Closing on a silhouette image, and to find the 8-directional chain "
        "code of the object boundary.",
        [("A. Morphological Operations", [
            "Read the silhouette image and convert it to a binary image.",
            "Add some small white dots and black holes to act as noise.",
            "Create a 7 x 7 elliptical structuring element.",
            "Erosion: a pixel stays white only if the whole structuring element fits inside the object.",
            "Dilation: a pixel becomes white if the structuring element touches the object.",
            "Opening = Erosion followed by Dilation. Closing = Dilation followed by Erosion.",
            "Apply Opening and then Closing to get a clean silhouette.",
        ]), ("B. Chain Code", [
            "Keep the largest connected object.",
            "Trace the outer boundary of the object pixel by pixel.",
            "For each move to the next boundary pixel, write the direction number (0 = East, 1 = North-East, "
            "2 = North, ... 7 = South-East).",
            "Rebuild the boundary from the chain code to check that it is correct.",
        ])],
        F, "unit3_morphology_chaincode.py",
        [(o("input_silhouette.png"), "Input silhouette image (horse). Source: scikit-image sample data", dict(width=CONTENT_W * 0.6)),
         (o("input_noisy_binary.png"), "Binary image with added noise", dict(width=CONTENT_W * 0.6))],
        [(o("fig1_morphology.png"), "Erosion, Dilation, Opening and Closing", {}),
         (o("fig2_zoom.png"), "Zoomed view of the head region", {}),
         (o("fig0_directions.png"), "8-direction chain code numbering", dict(width=CONTENT_W * 0.38)),
         (o("fig3_chaincode.png"), "Object boundary and chain code direction count", {})],
        [table([["Image", "White pixels", "Objects", "Holes"],
                ["Original", "43,412", "1", "1"], ["Noisy input", "43,795", "102", "50"],
                ["Erosion", "33,835", "6", "24"], ["Dilation", "56,008", "79", "3"],
                ["Opening", "42,452", "1", "34"], ["Closing", "44,923", "91", "2"],
                ["Opening + Closing", "43,376", "1", "2"]],
               [48 * mm, 32 * mm, 26 * mm, 22 * mm]),
         table([["Chain code result", "Value"],
                ["Starting point (x, y)", "(344, 15)"], ["Length of chain code", "1620"],
                ["Perimeter", "1833.7 pixels"], ["Boundary rebuilt from code", "Correct"],
                ["First 40 codes", "5455454444444544444444544445444544545454"]],
               [55 * mm, 85 * mm])],
        ["Erosion made the object smaller and removed the noise dots outside it, but the holes inside became bigger.",
         "Dilation made the object bigger and filled the holes, but the noise dots outside also grew.",
         "Opening removed all the outside noise (102 objects reduced to 1). Closing filled the holes inside "
         "(50 holes reduced to 2).",
         "Opening followed by Closing gave a clean shape with almost the same area as the original.",
         "In the chain code, directions 2 (North) and 6 (South) occur most often because of the long vertical legs."],
        "Erosion and dilation have opposite effects: one shrinks the object and the other grows it. Opening removes "
        "small noise and closing fills small holes, so together they gave a clean silhouette. The 8-directional chain "
        "code described the boundary completely, and the original boundary could be rebuilt from it.",
        "Unit3_Binary_Shape_Analysis.pdf")


# =====================================================================
def unit4():
    F = "Unit4_Image_Segmentation"
    o = lambda f: p(F, "outputs", f)
    standard(
        4, "Image Segmentation",
        "Implement region segmentation using Otsu's Adaptive Thresholding and Watershed Segmentation to cleanly "
        "isolate distinct foreground objects from highly varied background lighting conditions.",
        "To separate the coins from the background in an image with uneven lighting using Otsu's adaptive "
        "thresholding, and to separate the touching coins using Watershed segmentation.",
        [("A. Otsu's Thresholding", [
            "Read the coins image and add uneven lighting and a shadow to it.",
            "Smooth the image with a Gaussian filter.",
            "Global Otsu: find the one threshold that best separates the histogram into two classes "
            "(maximum between-class variance).",
            "Adaptive Otsu: find a separate Otsu threshold for a small circular region around every pixel.",
            "Clean the binary mask using morphological opening.",
        ]), ("B. Watershed Segmentation", [
            "Find the sure background by dilating the mask.",
            "Find the distance transform of the mask. The peaks are the centres of the coins and are used as markers.",
            "Mark the region between the background and the markers as unknown.",
            "Apply the watershed algorithm. Boundaries are drawn where the regions from two markers meet.",
            "Count the coins and measure their areas.",
        ])],
        F, "unit4_segmentation.py",
        [(o("input.png"), "Input image - 24 touching coins with added uneven lighting. Source: OpenCV sample image", dict(max_h=120 * mm))],
        [(o("fig1_otsu.png"), "Global Otsu and adaptive Otsu thresholding", dict(max_h=150 * mm)),
         (o("fig2_histogram.png"), "Histogram with the global Otsu threshold", dict(width=CONTENT_W * 0.9)),
         (o("fig3_watershed.png"), "Steps of watershed segmentation", dict(max_h=150 * mm)),
         (o("watershed_result.png"), "Final result - each coin separated and numbered", dict(max_h=125 * mm))],
        [table([["Method", "Result"],
                ["Global Otsu (T = 141)", "Dark background in the shadow is wrongly taken as coins"],
                ["Adaptive Otsu", "Correct coin mask, but touching coins are joined (1 region)"],
                ["Markers found", "24"],
                ["Watershed segmentation", "24 coins separated (actual = 24)"],
                ["Coin area", "6565 to 7396 pixels (average 6972)"]],
               [55 * mm, 95 * mm])],
        ["Global Otsu does not work here because the background in the dark corner is darker than the coins in the "
         "bright corner, so one threshold cannot separate them.",
         "Adaptive Otsu uses a different threshold for each region, so it handles the uneven lighting and gives a "
         "correct mask.",
         "Thresholding alone cannot separate coins that touch each other.",
         "Watershed, using the distance transform peaks as markers, separated all 24 coins.",
         "All the coins have similar areas, which shows that the boundaries are accurate."],
        "Global Otsu thresholding fails when the lighting is uneven. Adaptive Otsu thresholding solves this by using a "
        "local threshold. Watershed segmentation then separated the touching coins, and all 24 coins were correctly "
        "isolated.",
        "Unit4_Image_Segmentation.pdf")


UNITS = {1: unit1, 2: unit2, 3: unit3, 4: unit4}
if os.path.exists(p("build_unit5.py")):
    from build_unit5 import unit5
    UNITS[5] = lambda: unit5(standard, p)

if __name__ == "__main__":
    for u in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else UNITS):
        UNITS[u]()
