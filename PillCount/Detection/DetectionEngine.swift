//
//  DetectionEngine.swift
//  PillCount
//
//  Pulls pixel buffers off the camera queue, runs the OpenCV pipeline on a
//  dedicated serial queue (dropping frames while busy so the preview never
//  stalls), and reports results on the main thread.
//

import CoreImage
import CoreVideo
import Foundation
import UIKit

final class DetectionEngine {
    /// Called on the main thread with each processed frame.
    var onFrame: ((DetectionFrame) -> Void)?

    private let detector: PCPillDetector
    private let queue = DispatchQueue(label: "pillcount.detection", qos: .userInitiated)
    private let ciContext = CIContext()

    private let stateLock = NSLock()
    private var busy = false
    private var paused = false
    // The last buffer/result pair processed together, kept for freeze-frame
    // capture so the frozen image and its overlay always correspond.
    private var lastProcessedBuffer: CVPixelBuffer?
    private var lastProcessedFrame: DetectionFrame?

    init(params: PCDetectorParams = PCDetectorParams()) {
        detector = PCPillDetector(params: params)
    }

    /// Stop/resume consuming camera frames (used while frozen).
    func setPaused(_ value: Bool) {
        stateLock.lock()
        paused = value
        stateLock.unlock()
    }

    /// Entry point from the camera queue. Drops the frame if a detection is
    /// already in flight, keeping latency low without queue buildup.
    func process(_ pixelBuffer: CVPixelBuffer) {
        stateLock.lock()
        if busy || paused {
            stateLock.unlock()
            return
        }
        busy = true
        stateLock.unlock()

        queue.async { [weak self] in
            guard let self else { return }
            let result = self.detector.detect(in: pixelBuffer)
            let frame = DetectionFrame(
                pills: result.pills.enumerated().map { index, pill in
                    DetectedPill(
                        id: index,
                        contour: pill.contour.map { $0.cgPointValue },
                        center: pill.center)
                },
                imageSize: CGSize(
                    width: CVPixelBufferGetWidth(pixelBuffer),
                    height: CVPixelBufferGetHeight(pixelBuffer)),
                processingMillis: result.processingMillis,
                timestamp: Date())

            self.stateLock.lock()
            self.lastProcessedBuffer = pixelBuffer
            self.lastProcessedFrame = frame
            self.busy = false
            self.stateLock.unlock()

            DispatchQueue.main.async {
                self.onFrame?(frame)
            }
        }
    }

    /// Snapshot of the most recently processed frame, rendered to a UIImage,
    /// with its matching detection result. Used by the freeze button.
    func captureLastFrame() -> (image: UIImage, frame: DetectionFrame)? {
        stateLock.lock()
        let buffer = lastProcessedBuffer
        let frame = lastProcessedFrame
        stateLock.unlock()
        guard let buffer, let frame else { return nil }

        let ciImage = CIImage(cvPixelBuffer: buffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent)
        else { return nil }
        return (UIImage(cgImage: cgImage), frame)
    }
}
