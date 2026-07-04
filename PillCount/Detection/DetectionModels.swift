//
//  DetectionModels.swift
//  PillCount
//
//  Swift-side value types for detection results and count state.
//

import CoreGraphics
import Foundation

/// One detected pill, in normalized [0, 1] frame coordinates.
struct DetectedPill: Identifiable {
    let id: Int
    let contour: [CGPoint]
    let center: CGPoint
}

/// The result of processing one camera frame.
struct DetectionFrame {
    let pills: [DetectedPill]
    /// Pixel size of the (post-rotation) camera frame; needed to map
    /// normalized contours onto the aspect-fill preview.
    let imageSize: CGSize
    let processingMillis: Double
    let timestamp: Date
    /// True when `pills` came from the Core ML detector (classical OpenCV
    /// then ran as a cross-check); false when classical was the only path.
    var usedML: Bool = false
    /// The classical pipeline's count when ML was primary — shown to the
    /// user whenever the two detectors disagree.
    var crossCheckCount: Int? = nil

    var count: Int { pills.count }

    /// Non-nil disagreement between the two detectors (ML mode only).
    var crossCheckDelta: Int? {
        guard usedML, let cv = crossCheckCount, cv != pills.count else {
            return nil
        }
        return pills.count - cv
    }
}

/// The user-facing counting state produced by `CountSmoother`.
enum CountState: Equatable {
    /// No frames processed yet (camera starting, or permission missing).
    case searching
    /// A count is displayed but has not been stable long enough to trust.
    case counting(Int)
    /// The count has been identical for `lockFrames` consecutive frames.
    case locked(Int)

    var displayCount: Int? {
        switch self {
        case .searching: return nil
        case .counting(let n), .locked(let n): return n
        }
    }

    var isLocked: Bool {
        if case .locked = self { return true }
        return false
    }
}
