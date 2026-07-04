//
//  MLPillDetector.swift
//  PillCount
//
//  Optional Core ML detector path. If a compiled Core ML model named
//  "PillDetector" is present in the app bundle, it becomes the PRIMARY
//  detector (running on the Neural Engine) and the classical OpenCV
//  pipeline runs as a cross-check; with no model bundled the app works
//  exactly as before, classical-only. Produce the model with
//  tools/train_pill_yolo.py (a YOLO detector exported via
//  `format="coreml", nms=True`, which Vision consumes directly as an
//  object detector).
//
//  To install a model: drag PillDetector.mlpackage into Xcode, tick the
//  PillCount target. Nothing else — this class discovers it at launch.
//

import CoreML
import CoreVideo
import Foundation
import Vision

final class MLPillDetector {
    /// Base name of the bundled model (Xcode compiles .mlpackage to
    /// .mlmodelc inside the app bundle).
    static let modelResourceName = "PillDetector"

    /// Detections below this confidence are ignored. Tune against the
    /// calibration log: raise if phantom boxes appear, lower if faint
    /// pills are missed.
    var confidenceThreshold: Float = 0.35

    private let vnModel: VNCoreMLModel

    /// Fails (returns nil) when no model is bundled — callers treat that
    /// as "ML path unavailable" and stay classical-only.
    init?() {
        guard let url = Bundle.main.url(
            forResource: Self.modelResourceName, withExtension: "mlmodelc")
        else { return nil }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all   // prefer the Neural Engine
            let model = try MLModel(contentsOf: url, configuration: config)
            vnModel = try VNCoreMLModel(for: model)
        } catch {
            return nil
        }
    }

    /// Detect pills in a camera pixel buffer. Runs synchronously on the
    /// caller's (detection) queue. Returns nil on inference failure so the
    /// caller can fall back to the classical result.
    func detect(in pixelBuffer: CVPixelBuffer) -> [DetectedPill]? {
        let request = VNCoreMLRequest(model: vnModel)
        // YOLO Core ML exports are trained on square letterboxed input;
        // scaleFill matches how tools/train_pill_yolo.py exports the model.
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .up)
        do {
            try handler.perform([request])
        } catch {
            return nil
        }
        guard let observations = request.results as? [VNRecognizedObjectObservation]
        else { return nil }

        return observations
            .filter { $0.confidence >= confidenceThreshold }
            .enumerated()
            .map { index, obs in
                // Vision uses a bottom-left origin; our overlay space is
                // top-left. Flip Y and emit the box as a 4-point contour.
                let r = obs.boundingBox
                let top = 1.0 - r.maxY
                let bottom = 1.0 - r.minY
                let contour = [
                    CGPoint(x: r.minX, y: top),
                    CGPoint(x: r.maxX, y: top),
                    CGPoint(x: r.maxX, y: bottom),
                    CGPoint(x: r.minX, y: bottom),
                ]
                return DetectedPill(
                    id: index,
                    contour: contour,
                    center: CGPoint(x: r.midX, y: (top + bottom) / 2))
            }
    }
}
