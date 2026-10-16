"""Report content for Unit 5 (used by build_reports.py)."""
from reportlab.lib.units import mm
from report_engine import table, CONTENT_W


def unit5(standard, p):
    F = "Unit5_Object_Recognition"
    o = lambda f: p(F, "outputs", f)
    standard(
        5, "Modern Object Recognition",
        "Real-world inference application. Load a pre-trained deep learning framework (such as YOLO or MobileNet-SSD) "
        "to run real-time multi-class object detection and draw boundary tracking boxes with confidence scores over a "
        "custom input video track or webcam stream.",
        "To load a pre-trained YOLO model and use it to detect and track different types of objects in a video in real "
        "time, drawing a bounding box with the class name, track ID and confidence score for each object.",
        [("", [
            "Load the pre-trained YOLO11n model. It is trained on the COCO dataset and can detect 80 classes.",
            "Open the input video file (or the webcam, using index 0).",
            "Read the video one frame at a time.",
            "Pass the frame through the YOLO network. In a single pass it predicts the bounding boxes, class labels "
            "and confidence scores.",
            "Remove the boxes with confidence below 0.35. Duplicate boxes are removed by Non-Maximum Suppression.",
            "Use the ByteTrack tracker to match boxes between frames, so each object keeps the same track ID.",
            "Draw each box with its class name, track ID and confidence score, and show the FPS.",
            "Display or save the frame, and repeat until the video ends (or 'q' is pressed).",
            "Print the number of detections per class and the processing speed.",
        ])],
        F, "unit5_yolo_detection.py",
        [(o("input_frame.jpg"), "Input video (first frame) - street crossing, 854 x 480, 30 fps, 31 seconds. "
          "Source: Wikimedia Commons, by RG72, CC BY-SA 4.0", {})],
        [(o("frame_0222.jpg"), "Detection output - frame 222", {}),
         (o("frame_0514.jpg"), "Detection output - frame 514", {}),
         (o("timeline.png"), "Number of objects of each class detected over the video", {})],
        [table([["Item", "Value"],
                ["Model", "YOLO11n (pre-trained on COCO, 80 classes)"],
                ["Hardware", "NVIDIA GeForce RTX 4050 Laptop GPU"],
                ["Frames processed", "917"],
                ["Detection + tracking time", "6.8 ms per frame (146 FPS)"],
                ["Overall speed (read, detect, draw, save)", "87 FPS"]],
               [70 * mm, 80 * mm]),
         table([["Class", "Detections", "Mean confidence", "Track IDs"],
                ["person", "4308", "0.536", "109"], ["car", "3825", "0.652", "138"],
                ["traffic light", "1350", "0.577", "22"], ["handbag", "656", "0.486", "4"],
                ["backpack", "1", "0.391", "1"], ["umbrella", "1", "0.364", "1"]],
               [40 * mm, 32 * mm, 38 * mm, 30 * mm])],
        ["The program ran at 146 FPS (87 FPS including reading, drawing and saving). This is faster than the 30 fps "
         "video, so the detection works in real time.",
         "Four classes were detected in almost every frame: person, car, traffic light and handbag. Cars had the "
         "highest average confidence (0.65).",
         "Nearby objects were detected with high confidence (up to 0.94). Small and distant people and cars got lower "
         "scores, between 0.35 and 0.5.",
         "Backpack and umbrella were detected only once, with low confidence. These are most likely wrong detections "
         "and would be removed by a threshold of 0.5.",
         "There are more track IDs than real objects (for example 109 for person). A new ID is given when a person is "
         "hidden behind another object or is too far away to be detected for a few frames."],
        "A pre-trained YOLO11n model detected and tracked people, cars, traffic lights and handbags in a street video "
        "in real time, at 146 FPS on the GPU. Each object was shown with a bounding box, track ID and confidence score. "
        "Pre-trained deep learning models can be used for real-world detection without any training. However, small "
        "or distant objects get lower confidence, and the confidence threshold must be chosen carefully to remove "
        "false detections.",
        "Unit5_Modern_Object_Recognition.pdf")
